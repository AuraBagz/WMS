"""
WaterSlayer - Video Inpainter

Uses ProPainter for high-quality video inpainting.
Falls back to simpler methods if ProPainter is not available.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Callable
import subprocess
import tempfile
import shutil
import threading


# Add ProPainter to path
PROPAINTER_DIR = Path(__file__).parent.parent / "ProPainter"
if PROPAINTER_DIR.exists():
    sys.path.insert(0, str(PROPAINTER_DIR))


class _FFmpegWriter:
    """Streaming FFmpeg writer with background stderr drain to prevent deadlock."""

    def __init__(self, proc, stderr_thread, stderr_lines, output_path):
        self._proc = proc
        self._stderr_thread = stderr_thread
        self._stderr_lines = stderr_lines
        self._output_path = output_path
        self._closed = False

    def write_frame(self, frame):
        if frame is not None and not self._closed:
            try:
                self._proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                self._closed = True

    def close(self):
        if self._closed and self._proc.poll() is None:
            self._proc.kill()
        else:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            self._proc.wait()
        self._stderr_thread.join(timeout=5)

        if self._proc.returncode != 0:
            stderr = b''.join(self._stderr_lines).decode(errors='replace')
            print(f"[Inpainter] FFmpeg encoding failed (rc={self._proc.returncode}): {stderr[:500]}")
            raise RuntimeError(f"FFmpeg failed: {stderr[:200]}")
        else:
            print(f"[Inpainter] FFmpeg encoding complete: {self._output_path}")


class _OpenCVWriter:
    """OpenCV VideoWriter with same interface as _FFmpegWriter."""

    def __init__(self, writer):
        self._writer = writer

    def write_frame(self, frame):
        if frame is not None:
            self._writer.write(frame)

    def close(self):
        self._writer.release()


class VideoInpainter:
    """
    Video inpainting using ProPainter or fallback methods.

    ProPainter provides state-of-the-art video inpainting with
    temporal consistency.
    """

    def __init__(self, propainter_path: str = None):
        """
        Initialize inpainter.

        Args:
            propainter_path: Path to ProPainter installation (optional)
        """
        self.propainter_path = propainter_path or str(PROPAINTER_DIR)
        self.device = 'cuda' if self._check_cuda() else 'cpu'
        self.gpu_name = self._get_gpu_name()
        self.propainter_available = self._check_propainter()
        self._ffmpeg_exe = self._resolve_ffmpeg_exe()

        print(f"[Inpainter] Device: {self.device}")
        print(f"[Inpainter] GPU: {self.gpu_name}")
        print(f"[Inpainter] ProPainter path: {self.propainter_path}")
        print(f"[Inpainter] ProPainter available: {self.propainter_available}")
        print(f"[Inpainter] FFmpeg available: {self._ffmpeg_exe is not None}")
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def _get_gpu_name(self) -> str:
        """Get the active GPU name."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except Exception:
            pass
        return "CPU"

    def _is_rtx_5090(self) -> bool:
        return "5090" in self.gpu_name.lower()

    @staticmethod
    def _resolve_ffmpeg_exe() -> Optional[str]:
        """Find FFmpeg executable from system PATH or imageio-ffmpeg."""
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        try:
            import imageio_ffmpeg  # type: ignore
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
        return None

    @staticmethod
    def _resolve_ffprobe_exe() -> Optional[str]:
        """Find ffprobe executable from system PATH or derive from imageio-ffmpeg."""
        exe = shutil.which("ffprobe")
        if exe:
            return exe
        # Try to derive from ffmpeg path (ffprobe is usually next to ffmpeg)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            ffprobe_path = Path(ffmpeg).parent / ("ffprobe" + Path(ffmpeg).suffix)
            if ffprobe_path.exists():
                return str(ffprobe_path)
        try:
            import imageio_ffmpeg  # type: ignore
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            ffprobe_path = Path(ffmpeg_path).parent / ("ffprobe" + Path(ffmpeg_path).suffix)
            if ffprobe_path.exists():
                return str(ffprobe_path)
        except Exception:
            pass
        return None

    def _probe_video_bitrate(self, video_path: str) -> Optional[int]:
        """
        Get video stream bitrate in bits/s from input video using ffprobe.

        Returns bitrate as int, or None if unavailable.
        """
        ffprobe_exe = self._resolve_ffprobe_exe()
        if not ffprobe_exe:
            return None
        try:
            cmd = [
                ffprobe_exe, '-v', 'quiet',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=bit_rate',
                '-of', 'csv=p=0',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                val = result.stdout.strip().split('\n')[0].strip()
                if val and val != 'N/A':
                    return int(val)
        except Exception as e:
            print(f"[Inpainter] ffprobe bitrate query failed: {e}")

        # Fallback: try format-level bit_rate (total stream) and subtract audio estimate
        try:
            cmd = [
                ffprobe_exe, '-v', 'quiet',
                '-show_entries', 'format=bit_rate',
                '-of', 'csv=p=0',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                val = result.stdout.strip().split('\n')[0].strip()
                if val and val != 'N/A':
                    # Subtract ~192k for audio as rough estimate
                    return max(500_000, int(val) - 192_000)
        except Exception as e:
            print(f"[Inpainter] ffprobe format bitrate query failed: {e}")

        return None

    def _open_video_writer(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        source_video_path: str = None
    ):
        """
        Open a streaming FFmpeg video writer. Returns a context object with
        .write_frame(frame) and .close() methods.

        Uses H.264 with bitrate matching (probed from source) or CRF 16 fallback.
        Falls back to OpenCV VideoWriter if FFmpeg is unavailable.
        """
        ffmpeg_exe = self._ffmpeg_exe
        if not ffmpeg_exe:
            print("[Inpainter] FFmpeg not available, falling back to OpenCV VideoWriter")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            return _OpenCVWriter(out)

        # Probe input bitrate to match output quality
        input_bitrate = None
        if source_video_path:
            input_bitrate = self._probe_video_bitrate(source_video_path)
            if input_bitrate:
                print(f"[Inpainter] Input video bitrate: {input_bitrate // 1000} kbps — matching on output")

        cmd = [
            ffmpeg_exe, '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-pix_fmt', 'yuv420p',
        ]

        # Always use CRF for quality-based encoding.
        # Bitrate matching can degrade quality on re-encode (decode→process→encode
        # generation loss), especially on lower-bitrate source files.
        if input_bitrate:
            print(f"[Inpainter] Source bitrate {input_bitrate // 1000} kbps noted; using CRF 10 for quality preservation")
        cmd += ['-crf', '10']

        cmd.append(output_path)

        print(f"[Inpainter] Opening FFmpeg H.264 encoder (streaming)...")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        # Drain stderr in background thread to prevent pipe deadlock
        stderr_lines = []
        def _drain_stderr():
            for line in proc.stderr:
                stderr_lines.append(line)
        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        return _FFmpegWriter(proc, stderr_thread, stderr_lines, output_path)

    def _write_frames_to_video(
        self,
        frames,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        source_video_path: str = None
    ):
        """Write a list of frames to video. Convenience wrapper around _open_video_writer."""
        writer = self._open_video_writer(output_path, fps, width, height, source_video_path)
        try:
            for frame in frames:
                writer.write_frame(frame)
        finally:
            writer.close()

    def _resolve_quality_mode(self, quality_mode: str) -> str:
        if quality_mode == 'auto':
            return 'rtx5090' if self._is_rtx_5090() else 'balanced'
        return quality_mode

    def _quality_profile(self, quality_mode: str) -> dict:
        mode = self._resolve_quality_mode(quality_mode)

        if mode == 'rtx5090':
            return {
                "name": "RTX 5090 Maximum Quality (ProPainter-aligned)",
                "chunk_size": 64,
                "overlap": 12,
                "neighbor_length": 10,
                "ref_stride": 10,
                "raft_iter": 20,
                "mask_dilation": 4,
                "subvideo_length": 64,
                "use_fp16": True
            }

        if mode == 'rtx5090_crisp':
            return {
                "name": "RTX 5090 Crisp (Tight Mask)",
                "chunk_size": 56,
                "overlap": 10,
                "neighbor_length": 8,
                "ref_stride": 10,
                "raft_iter": 20,
                "mask_dilation": 2,
                "subvideo_length": 48,
                "use_fp16": True
            }

        return {
            "name": "Balanced",
            "chunk_size": 40,
            "overlap": 10,
            "neighbor_length": 10,
            "ref_stride": 10,
            "raft_iter": 20,
            "mask_dilation": 4,
            "subvideo_length": 40,
            "use_fp16": True
        }

    def _is_memory_error(self, message: str) -> bool:
        text = (message or "").lower()
        patterns = [
            "out of memory",
            "outofmemoryerror",
            "cuda out of memory",
            "host_allocation_failed",
            "cudnn_status_internal_error_host_allocation_failed",
            "cuda error: out of memory",
        ]
        return any(p in text for p in patterns)

    def _is_propainter_capacity_error(self, message: str) -> bool:
        """
        Detect ProPainter/RAFT failures caused by tensor size/capacity limits.
        These should trigger safer retry settings instead of hard-failing.
        """
        text = (message or "").lower()
        if self._is_memory_error(text):
            return True
        patterns = [
            "integer out of range",
            "cudnn_status_not_supported",
            "cublas_status_not_supported",
            "cuda error: invalid configuration argument",
            "resource exhausted",
        ]
        return any(p in text for p in patterns)
    
    def _check_propainter(self) -> bool:
        """Check if ProPainter is available."""
        propainter_dir = Path(self.propainter_path)
        
        # Check if directory and required files exist
        if not propainter_dir.exists():
            print(f"[Inpainter] ProPainter dir not found: {propainter_dir}")
            return False
        
        inference_script = propainter_dir / "inference_propainter.py"
        if not inference_script.exists():
            print(f"[Inpainter] inference_propainter.py not found")
            return False
        
        # Check for weights
        weights_dir = propainter_dir / "weights"
        required_weights = ["ProPainter.pth", "recurrent_flow_completion.pth", "raft-things.pth"]
        
        for weight in required_weights:
            if not (weights_dir / weight).exists():
                print(f"[Inpainter] Missing weight: {weight}")
                return False
        
        return True
    
    def _mux_audio_to_video(self, video_no_audio: str, original_video: str, output_path: str) -> str:
        """
        Mux audio from original video into the processed video using FFmpeg.

        Tries stream-copying audio first (zero quality loss), falls back to
        AAC re-encoding if the container doesn't support the original codec.

        Args:
            video_no_audio: Path to processed video (no audio)
            original_video: Path to original video with audio
            output_path: Final output path with audio

        Returns:
            Path to output video with audio
        """
        print(f"[Inpainter] Muxing audio from original video...")

        ffmpeg_exe = self._ffmpeg_exe
        if not ffmpeg_exe:
            print("[Inpainter] FFmpeg not found (PATH/imageio-ffmpeg), output will have no audio")
            shutil.copy(video_no_audio, output_path)
            return output_path

        # Try 1: Stream-copy audio (zero quality loss)
        cmd_copy = [
            ffmpeg_exe, '-y',
            '-i', video_no_audio,
            '-i', original_video,
            '-c:v', 'copy',
            '-c:a', 'copy',            # Stream-copy audio (no re-encoding)
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-shortest',
            output_path
        ]

        try:
            result = subprocess.run(cmd_copy, capture_output=True, text=True, timeout=300)

            if result.returncode == 0 and Path(output_path).exists():
                print(f"[Inpainter] Audio stream-copied successfully: {output_path}")
                if Path(video_no_audio).exists() and video_no_audio != output_path:
                    os.remove(video_no_audio)
                return output_path
            else:
                print(f"[Inpainter] Audio stream-copy failed, retrying with AAC re-encode...")
        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"[Inpainter] Audio stream-copy error: {e}, retrying with AAC re-encode...")

        # Try 2: Re-encode audio as AAC (fallback for incompatible codecs)
        cmd_aac = [
            ffmpeg_exe, '-y',
            '-i', video_no_audio,
            '-i', original_video,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-shortest',
            output_path
        ]

        try:
            result = subprocess.run(cmd_aac, capture_output=True, text=True, timeout=300)

            if result.returncode == 0 and Path(output_path).exists():
                print(f"[Inpainter] Audio muxed (AAC fallback) successfully: {output_path}")
                if Path(video_no_audio).exists() and video_no_audio != output_path:
                    os.remove(video_no_audio)
                return output_path
            else:
                print(f"[Inpainter] FFmpeg muxing failed: {result.stderr[:500]}")
                shutil.copy(video_no_audio, output_path)
                return output_path

        except subprocess.TimeoutExpired:
            print("[Inpainter] FFmpeg muxing timed out")
            shutil.copy(video_no_audio, output_path)
            return output_path
        except Exception as e:
            print(f"[Inpainter] Audio muxing error: {e}")
            shutil.copy(video_no_audio, output_path)
            return output_path
    
    def inpaint_video(
        self,
        video_path: str,
        masks: Dict[int, np.ndarray],
        output_path: str,
        method: str = 'auto',
        quality_mode: str = 'balanced',
        detail_restore_mode: str = 'off',
        progress_callback: Callable = None
    ) -> str:
        """
        Inpaint video with given masks.
        
        Args:
            video_path: Input video path
            masks: Dict mapping frame_index -> binary mask (255 = inpaint)
            output_path: Output video path
            method: 'propainter', 'opencv', or 'auto'
            quality_mode: 'balanced', 'rtx5090', 'rtx5090_crisp', or 'auto'
            detail_restore_mode: 'off', 'roi_sharpen', or 'roi_sharpen_strong'
            progress_callback: Called with (current, total, message)
            
        Returns:
            Path to output video
        """
        if method == 'auto':
            method = 'propainter' if self.propainter_available else 'opencv'
        
        print(f"[Inpainter] Using method: {method}")
        
        if method == 'propainter':
            if not self.propainter_available:
                print("[Inpainter] ProPainter not available, falling back to OpenCV")
                return self._inpaint_opencv(video_path, masks, output_path, progress_callback)
            return self._inpaint_propainter(
                video_path,
                masks,
                output_path,
                quality_mode,
                detail_restore_mode,
                progress_callback
            )
        else:
            return self._inpaint_opencv(video_path, masks, output_path, progress_callback)

    def _apply_roi_detail_restore(
        self,
        original_frame: np.ndarray,
        inpainted_frame: np.ndarray,
        mask: np.ndarray,
        mode: str
    ) -> np.ndarray:
        """Restore texture detail only inside/around inpainted ROI."""
        if mode == 'off' or mask is None or not np.any(mask):
            return inpainted_frame

        if mode == 'roi_sharpen_strong':
            amount = 1.25
            sigma = 1.4
            radius = 9
        else:
            amount = 0.85
            sigma = 1.1
            radius = 7

        # Build a soft ROI blend mask from the binary inpaint mask.
        roi = (mask > 127).astype(np.uint8) * 255
        roi = cv2.dilate(roi, np.ones((3, 3), np.uint8), iterations=1)
        roi_soft = cv2.GaussianBlur(roi.astype(np.float32) / 255.0, (radius, radius), sigma)
        roi_soft = np.clip(roi_soft, 0.0, 1.0)
        roi_soft_3 = np.repeat(roi_soft[:, :, None], 3, axis=2)

        # Unsharp only on ProPainter output, then blend back into output in ROI.
        blur = cv2.GaussianBlur(inpainted_frame, (0, 0), sigma)
        sharp = cv2.addWeighted(inpainted_frame, 1.0 + amount, blur, -amount, 0.0)
        sharp = np.clip(sharp, 0, 255).astype(np.uint8)

        out = (
            (1.0 - roi_soft_3) * inpainted_frame.astype(np.float32)
            + roi_soft_3 * sharp.astype(np.float32)
        )

        # Keep transition slightly anchored to the original frame to avoid halos.
        edge_band = cv2.Canny(roi, 30, 90)
        if np.any(edge_band):
            edge = cv2.GaussianBlur(edge_band.astype(np.float32) / 255.0, (9, 9), 2.0)
            edge3 = np.repeat(edge[:, :, None], 3, axis=2) * 0.15
            out = (
                (1.0 - edge3) * out
                + edge3 * original_frame.astype(np.float32)
            )

        return np.clip(out, 0, 255).astype(np.uint8)
    
    def _inpaint_opencv(
        self,
        video_path: str,
        masks: Dict[int, np.ndarray],
        output_path: str,
        progress_callback: Callable = None
    ) -> str:
        """
        Enhanced OpenCV inpainting with seamless blending.
        
        Uses multi-pass inpainting and alpha blending for smoother results.
        """
        print("[Inpainter] Using enhanced OpenCV inpainting (frame-by-frame)")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Stream frames directly to encoder — no RAM accumulation
        writer = self._open_video_writer(output_path, fps, width, height, source_video_path=video_path)
        frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                mask = masks.get(frame_idx)

                if mask is not None and np.any(mask):
                    writer.write_frame(self._seamless_inpaint(frame, mask))
                else:
                    writer.write_frame(frame)

                if progress_callback:
                    progress_callback(frame_idx + 1, total_frames, "Inpainting (OpenCV)")

                frame_idx += 1
        finally:
            writer.close()

        cap.release()

        return output_path
    
    def _seamless_inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Perform seamless inpainting with alpha blending for natural results.
        """
        # Create expanded mask for better coverage
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=2)
        
        # First pass: Navier-Stokes inpainting with large radius
        inpainted = cv2.inpaint(frame, mask_dilated, 15, cv2.INPAINT_NS)
        
        # Second pass: Telea for texture
        inpainted2 = cv2.inpaint(frame, mask_dilated, 10, cv2.INPAINT_TELEA)
        
        # Blend both methods
        inpainted = cv2.addWeighted(inpainted, 0.6, inpainted2, 0.4, 0)
        
        # Create soft alpha mask for seamless blending
        # Start with dilated mask, heavily blur for gradient
        blur_size = 51
        alpha = mask_dilated.astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (blur_size, blur_size), 25)
        
        # Expand alpha slightly at edges
        alpha = np.clip(alpha * 1.2, 0, 1)
        alpha = cv2.GaussianBlur(alpha, (31, 31), 15)
        
        # Make it 3-channel
        alpha = np.stack([alpha, alpha, alpha], axis=-1)
        
        # Blend: use inpainted where alpha is high, original where low
        frame_float = frame.astype(np.float32)
        inpainted_float = inpainted.astype(np.float32)
        
        result = (1 - alpha) * frame_float + alpha * inpainted_float
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Final subtle blur on the blend boundary to remove any remaining artifacts
        # Only apply to the transition zone
        transition_mask = ((alpha[:,:,0] > 0.05) & (alpha[:,:,0] < 0.95)).astype(np.uint8) * 255
        if np.any(transition_mask):
            transition_mask = cv2.dilate(transition_mask, np.ones((7, 7), np.uint8), iterations=1)
            
            # Create a subtle blur layer
            result_blurred = cv2.GaussianBlur(result, (5, 5), 2)
            
            # Blend at transition
            trans_alpha = transition_mask.astype(np.float32) / 255.0 * 0.3
            trans_alpha = np.stack([trans_alpha, trans_alpha, trans_alpha], axis=-1)
            
            result = ((1 - trans_alpha) * result.astype(np.float32) + 
                     trans_alpha * result_blurred.astype(np.float32))
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _inpaint_propainter(
        self,
        video_path: str,
        masks: Dict[int, np.ndarray],
        output_path: str,
        quality_mode: str = 'balanced',
        detail_restore_mode: str = 'off',
        progress_callback: Callable = None
    ) -> str:
        """
        High-quality inpainting using ProPainter with chunked processing.
        
        Processes video in overlapping chunks to:
        - Maintain full resolution (no downscaling)
        - Handle long videos within VRAM limits
        - Preserve temporal consistency via overlapping blends
        """
        print("[Inpainter] Using ProPainter with chunked processing (full quality)")
        print(f"[Inpainter] Detail restore mode: {detail_restore_mode}")
        profile = self._quality_profile(quality_mode)

        chunk_size = profile["chunk_size"]
        overlap = profile["overlap"]
        neighbor_length = profile["neighbor_length"]
        ref_stride = profile["ref_stride"]
        raft_iter = profile["raft_iter"]
        mask_dilation = profile["mask_dilation"]
        subvideo_length = profile["subvideo_length"]
        use_fp16 = profile["use_fp16"]

        print(f"[Inpainter] Quality profile: {profile['name']}")
        
        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp(prefix="waterslayer_"))
        
        try:
            # Extract all frames and masks first
            if progress_callback:
                progress_callback(0, 100, "Extracting frames...")
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"[Inpainter] Video: {width}x{height}, {total_frames} frames, {fps} fps")
            # Resolution-aware tuning:
            # keep ProPainter default temporal settings for <= 720p-ish inputs.
            # only trim very high resolutions where memory spikes are common.
            megapixels = (width * height) / 1_000_000.0
            if self._resolve_quality_mode(quality_mode) == "rtx5090" and megapixels >= 1.6:
                chunk_size = min(chunk_size, 56)
                overlap = min(overlap, 10)
                neighbor_length = min(neighbor_length, 10)
                ref_stride = max(ref_stride, 10)
                raft_iter = min(raft_iter, 20)
                subvideo_length = min(subvideo_length, 48)

            print(
                f"[Inpainter] Chunk size: {chunk_size}, Overlap: {overlap}, "
                f"neighbor={neighbor_length}, ref_stride={ref_stride}, "
                f"raft_iter={raft_iter}, subvideo={subvideo_length}, fp16={use_fp16}"
            )
            
            # Calculate number of chunks
            if total_frames <= chunk_size:
                num_chunks = 1
            else:
                effective_chunk = chunk_size - overlap
                num_chunks = max(1, (total_frames - overlap + effective_chunk - 1) // effective_chunk)
            
            print(f"[Inpainter] Processing in {num_chunks} chunk(s)")
            
            # Store all frames in memory for chunking
            all_frames = []
            all_masks = []
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                all_frames.append(frame)
                mask = masks.get(frame_idx, np.zeros((height, width), dtype=np.uint8))
                all_masks.append(mask)
                
                if progress_callback and frame_idx % 20 == 0:
                    progress = int((frame_idx / total_frames) * 10)
                    progress_callback(progress, 100, f"Loading frame {frame_idx}/{total_frames}")
                
                frame_idx += 1
            
            cap.release()
            total_frames = len(all_frames)
            
            # Process each chunk
            all_result_frames = [None] * total_frames
            
            for chunk_idx in range(num_chunks):
                # Calculate chunk boundaries
                start_frame = chunk_idx * (chunk_size - overlap)
                end_frame = min(start_frame + chunk_size, total_frames)
                chunk_frames = list(range(start_frame, end_frame))
                
                print(f"[Inpainter] Processing chunk {chunk_idx + 1}/{num_chunks}: frames {start_frame}-{end_frame-1}")
                
                if progress_callback:
                    base_progress = 10 + int((chunk_idx / num_chunks) * 80)
                    progress_callback(base_progress, 100, f"Chunk {chunk_idx + 1}/{num_chunks}")
                
                # Create temp dirs for this chunk
                chunk_dir = temp_dir / f"chunk_{chunk_idx}"
                chunk_frames_dir = chunk_dir / "frames"
                chunk_masks_dir = chunk_dir / "masks"
                chunk_output_dir = chunk_dir / "output"
                
                chunk_frames_dir.mkdir(parents=True)
                chunk_masks_dir.mkdir(parents=True)
                chunk_output_dir.mkdir(parents=True)
                
                # Write chunk frames and masks
                for local_idx, global_idx in enumerate(chunk_frames):
                    frame_path = chunk_frames_dir / f"{local_idx:05d}.png"
                    mask_path = chunk_masks_dir / f"{local_idx:05d}.png"
                    
                    cv2.imwrite(str(frame_path), all_frames[global_idx])
                    cv2.imwrite(str(mask_path), all_masks[global_idx])
                
                # Run ProPainter on this chunk
                try:
                    self._run_propainter_chunk(
                        chunk_frames_dir, 
                        chunk_masks_dir, 
                        chunk_output_dir,
                        width, 
                        height,
                        len(chunk_frames),
                        fps=fps,
                        neighbor_length=neighbor_length,
                        ref_stride=ref_stride,
                        raft_iter=raft_iter,
                        mask_dilation=mask_dilation,
                        subvideo_length=min(len(chunk_frames), subvideo_length),
                        use_fp16=use_fp16
                    )
                except RuntimeError as e:
                    # Retry this chunk with safer settings instead of hard-failing.
                    if not self._is_propainter_capacity_error(str(e)):
                        raise

                    retry_neighbor = max(6, neighbor_length - 2)
                    retry_ref_stride = max(12, ref_stride + 2)
                    retry_raft_iter = max(12, raft_iter - 4)
                    retry_subvideo = max(24, min(len(chunk_frames), subvideo_length - 16))
                    print(
                        f"[Inpainter] OOM on chunk {chunk_idx + 1}; retrying with safer settings "
                        f"(neighbor={retry_neighbor}, ref_stride={retry_ref_stride}, "
                        f"raft_iter={retry_raft_iter}, subvideo={retry_subvideo}, fp16=True)"
                    )

                    try:
                        self._run_propainter_chunk(
                            chunk_frames_dir, 
                            chunk_masks_dir, 
                            chunk_output_dir,
                            width, 
                            height,
                            len(chunk_frames),
                            fps=fps,
                            neighbor_length=retry_neighbor,
                            ref_stride=retry_ref_stride,
                            raft_iter=retry_raft_iter,
                            mask_dilation=mask_dilation,
                            subvideo_length=retry_subvideo,
                            use_fp16=True
                        )
                    except RuntimeError as e2:
                        if not self._is_propainter_capacity_error(str(e2)):
                            raise
                        # Last-chance retry: lower internal processing size for this chunk.
                        # For 4K this typically lands near ~1080p internal processing.
                        target_pixels = 1920 * 1080
                        frame_pixels = max(1, width * height)
                        dynamic_ratio = (target_pixels / float(frame_pixels)) ** 0.5
                        safe_ratio = min(0.9, max(0.45, dynamic_ratio))
                        print(
                            f"[Inpainter] Memory pressure persists on chunk {chunk_idx + 1}; "
                            f"retrying with resize_ratio={safe_ratio:.2f} fallback"
                        )
                        try:
                            self._run_propainter_chunk(
                                chunk_frames_dir,
                                chunk_masks_dir,
                                chunk_output_dir,
                                width,
                                height,
                                len(chunk_frames),
                                fps=fps,
                                neighbor_length=max(6, retry_neighbor - 1),
                                ref_stride=max(12, retry_ref_stride),
                                raft_iter=max(10, retry_raft_iter - 2),
                                mask_dilation=max(2, mask_dilation - 1),
                                subvideo_length=max(20, min(len(chunk_frames), retry_subvideo - 8)),
                                use_fp16=True,
                                resize_ratio=safe_ratio
                            )
                        except RuntimeError as e3:
                            if not self._is_propainter_capacity_error(str(e3)):
                                raise
                            # Final fallback for pathological clips/resolutions.
                            final_ratio = max(0.40, safe_ratio * 0.85)
                            print(
                                f"[Inpainter] Capacity error persists on chunk {chunk_idx + 1}; "
                                f"final retry with resize_ratio={final_ratio:.2f}"
                            )
                            self._run_propainter_chunk(
                                chunk_frames_dir,
                                chunk_masks_dir,
                                chunk_output_dir,
                                width,
                                height,
                                len(chunk_frames),
                                fps=fps,
                                neighbor_length=max(6, retry_neighbor - 2),
                                ref_stride=max(14, retry_ref_stride + 2),
                                raft_iter=max(8, retry_raft_iter - 4),
                                mask_dilation=max(2, mask_dilation - 2),
                                subvideo_length=max(16, min(len(chunk_frames), retry_subvideo - 12)),
                                use_fp16=True,
                                resize_ratio=final_ratio
                            )
                
                # ProPainter saves to output/video_name/inpaint_out.mp4
                # The video_name is the name of the frames directory (e.g., "frames")
                frames_dir_name = chunk_frames_dir.name  # This will be "frames"
                result_video = chunk_output_dir / frames_dir_name / "inpaint_out.mp4"
                
                print(f"[Inpainter] Looking for result at: {result_video}")
                
                if not result_video.exists():
                    # Try alternative paths
                    alt_paths = [
                        chunk_output_dir / "inpaint_out.mp4",
                        chunk_output_dir / "frames" / "inpaint_out.mp4",
                    ]
                    for alt in alt_paths:
                        if alt.exists():
                            result_video = alt
                            break
                
                # Prefer lossless PNG frames over the lossy inpaint_out.mp4
                frames_png_dir = chunk_output_dir / frames_dir_name / "frames"
                png_files = sorted(frames_png_dir.glob("*.png")) if frames_png_dir.exists() else []

                if png_files:
                    # Read lossless PNGs — no intermediate lossy encode
                    result_frames = [cv2.imread(str(p)) for p in png_files]
                    result_frames = [f for f in result_frames if f is not None]
                    print(f"[Inpainter] Read {len(result_frames)} lossless PNG frames from ProPainter output")
                elif result_video.exists():
                    # Fallback: extract frames from the compressed video
                    result_cap = cv2.VideoCapture(str(result_video))
                    result_frames = []
                    while True:
                        ret, frame = result_cap.read()
                        if not ret:
                            break
                        result_frames.append(frame)
                    result_cap.release()
                    print(f"[Inpainter] Extracted {len(result_frames)} frames from ProPainter video (lossy fallback)")
                else:
                    result_frames = []
                    print(f"[Inpainter] ERROR: ProPainter output not found!")
                    print(f"[Inpainter] Searched: {result_video}")
                    if chunk_output_dir.exists():
                        print(f"[Inpainter] Contents of {chunk_output_dir}:")
                        for item in chunk_output_dir.rglob("*"):
                            print(f"  - {item}")

                if result_frames:
                    if len(result_frames) != len(chunk_frames):
                        print(f"[Inpainter] Warning: Expected {len(chunk_frames)} results, got {len(result_frames)}")

                    # Store results with overlap blending
                    for local_idx, global_idx in enumerate(chunk_frames):
                        if local_idx < len(result_frames):
                            result_frame = result_frames[local_idx]

                            # Resize back to original if needed
                            if result_frame.shape[:2] != (height, width):
                                result_frame = cv2.resize(result_frame, (width, height),
                                                         interpolation=cv2.INTER_LANCZOS4)

                            # Handle overlap blending
                            if all_result_frames[global_idx] is not None:
                                overlap_position = local_idx
                                blend_alpha = overlap_position / overlap

                                prev_frame = all_result_frames[global_idx]
                                blended = cv2.addWeighted(
                                    prev_frame, 1 - blend_alpha,
                                    result_frame, blend_alpha,
                                    0
                                )
                                all_result_frames[global_idx] = blended
                            else:
                                all_result_frames[global_idx] = result_frame
                
                # Clean up chunk temp files
                shutil.rmtree(chunk_dir, ignore_errors=True)
            
            # Assemble final video — stream frames to encoder and free RAM as we go
            if progress_callback:
                progress_callback(92, 100, "Encoding final video...")

            temp_video_path = str(temp_dir / "output_no_audio.mp4")
            writer = self._open_video_writer(temp_video_path, fps, width, height, source_video_path=video_path)

            try:
                for idx in range(total_frames):
                    # Pick result frame or fall back to original
                    frame = all_result_frames[idx] if all_result_frames[idx] is not None else all_frames[idx]

                    # Optional detail restore
                    if detail_restore_mode != 'off' and all_result_frames[idx] is not None:
                        frame = self._apply_roi_detail_restore(
                            all_frames[idx], frame, all_masks[idx], detail_restore_mode
                        )

                    writer.write_frame(frame)

                    # Free memory — frame already written to encoder
                    all_result_frames[idx] = None
                    all_frames[idx] = None
                    all_masks[idx] = None

                    if progress_callback and idx % 100 == 0:
                        encode_pct = 92 + int((idx / total_frames) * 4)
                        progress_callback(encode_pct, 100, f"Encoding frame {idx}/{total_frames}")
            finally:
                writer.close()
            
            # Mux audio from original video
            if progress_callback:
                progress_callback(96, 100, "Adding audio...")
            
            self._mux_audio_to_video(temp_video_path, video_path, output_path)
            
            if progress_callback:
                progress_callback(100, 100, "Complete!")
            
            print(f"[Inpainter] ProPainter chunked processing complete: {output_path}")
            return output_path
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _run_propainter_chunk(
        self,
        frames_dir: Path,
        masks_dir: Path,
        output_dir: Path,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        neighbor_length: int,
        ref_stride: int,
        raft_iter: int,
        mask_dilation: int,
        subvideo_length: int,
        use_fp16: bool,
        resize_ratio: float = 1.0
    ):
        """Run ProPainter on a single chunk at FULL resolution."""
        propainter_dir = Path(self.propainter_path)
        inference_script = propainter_dir / "inference_propainter.py"
        
        python_exe = sys.executable
        
        print(f"[Inpainter] Processing chunk: {num_frames} frames at {width}x{height} (full resolution)")
        
        # FULL RESOLUTION - no downscaling
        # We handle memory by chunking, not by reducing quality
        cmd = [
            python_exe,
            str(inference_script),
            "--video", str(frames_dir),
            "--mask", str(masks_dir),
            "--output", str(output_dir),
            # Full resolution settings
            "--resize_ratio", str(resize_ratio),
            "--height", str(height),
            "--width", str(width),
            # Temporal parameters for quality
            "--neighbor_length", str(neighbor_length),
            "--ref_stride", str(ref_stride),
            "--raft_iter", str(raft_iter),
            "--mask_dilation", str(mask_dilation),
            "--subvideo_length", str(subvideo_length),
            "--save_fps", str(max(1, int(round(fps)))),
            "--save_frames",  # Save lossless PNGs to avoid lossy intermediate video
        ]

        if use_fp16:
            cmd.append("--fp16")
        
        print(f"[Inpainter] Running ProPainter with memory optimizations...")
        print(f"[Inpainter] Command: {' '.join(cmd)}")
        
        # Set environment variable for better memory management
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        
        result = subprocess.run(
            cmd,
            cwd=str(propainter_dir),
            capture_output=True,
            text=True,
            env=env
        )
        
        if result.returncode != 0:
            print(f"[Inpainter] ProPainter stderr: {result.stderr}")
            # If OOM, provide helpful message
            if self._is_memory_error(result.stderr):
                raise RuntimeError(
                    f"CUDA Out of Memory on chunk! Chunk size may need to be reduced. "
                    f"Current: {num_frames} frames at {width}x{height}"
                )
            raise RuntimeError(f"ProPainter failed: {result.stderr[:500]}")
        
        print(f"[Inpainter] Chunk completed successfully")
    
    def _frames_to_video(self, frames_dir: Path, output_path: str, fps: float):
        """Assemble frames into a video."""
        # Find all PNG files, sorted
        frame_files = sorted(frames_dir.glob("*.png"))
        
        if not frame_files:
            frame_files = sorted(frames_dir.glob("*.jpg"))
        
        if not frame_files:
            raise ValueError(f"No frames found in {frames_dir}")
        
        print(f"[Inpainter] Assembling {len(frame_files)} frames at {fps} fps")

        # Read first frame for dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        height, width = first_frame.shape[:2]

        # Stream frames directly to encoder — no RAM accumulation
        writer = self._open_video_writer(output_path, fps, width, height)
        try:
            writer.write_frame(first_frame)
            for frame_file in frame_files[1:]:
                frame = cv2.imread(str(frame_file))
                if frame is not None:
                    writer.write_frame(frame)
        finally:
            writer.close()
        print(f"[Inpainter] Video saved to {output_path}")


class SimpleInpainter:
    """
    Simpler inpainting for quick results.
    
    Uses telea/NS inpainting per frame - fast but may have artifacts.
    """
    
    @staticmethod
    def inpaint_frame(frame: np.ndarray, mask: np.ndarray, radius: int = 7) -> np.ndarray:
        """Inpaint a single frame."""
        if mask is None or not np.any(mask):
            return frame
        
        # Use Navier-Stokes based inpainting for better quality
        return cv2.inpaint(frame, mask, radius, cv2.INPAINT_NS)
    
    @staticmethod
    def inpaint_video_simple(
        video_path: str,
        masks: Dict[int, np.ndarray],
        output_path: str,
        progress_callback: Callable = None
    ) -> str:
        """Quick inpainting for previews."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        processed_frames = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mask = masks.get(frame_idx)
            if mask is not None and np.any(mask):
                frame = SimpleInpainter.inpaint_frame(frame, mask)

            processed_frames.append(frame)

            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx + 1, total_frames, "Quick inpaint")

            frame_idx += 1

        cap.release()

        # Use FFmpeg for quality-preserving encoding if available
        ffmpeg_exe = VideoInpainter._resolve_ffmpeg_exe()
        if ffmpeg_exe:
            # Probe input bitrate
            input_bitrate = None
            ffprobe_exe = VideoInpainter._resolve_ffprobe_exe()
            if ffprobe_exe:
                try:
                    cmd = [ffprobe_exe, '-v', 'quiet', '-select_streams', 'v:0',
                           '-show_entries', 'stream=bit_rate', '-of', 'csv=p=0', video_path]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0 and result.stdout.strip():
                        val = result.stdout.strip().split('\n')[0].strip()
                        if val and val != 'N/A':
                            input_bitrate = int(val)
                except Exception:
                    pass

            cmd = [
                ffmpeg_exe, '-y',
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}', '-pix_fmt', 'bgr24',
                '-r', str(fps), '-i', '-',
                '-c:v', 'libx264', '-preset', 'medium', '-pix_fmt', 'yuv420p',
            ]
            if input_bitrate:
                cmd += ['-b:v', str(input_bitrate)]
            else:
                cmd += ['-crf', '16']
            cmd.append(output_path)

            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            for frame in processed_frames:
                try:
                    proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    break
            proc.stdin.close()
            proc.wait()

            if proc.returncode == 0:
                return output_path

        # Fallback to OpenCV VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in processed_frames:
            out.write(frame)
        out.release()

        return output_path
