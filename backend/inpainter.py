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


# Add ProPainter to path
PROPAINTER_DIR = Path(__file__).parent.parent / "ProPainter"
if PROPAINTER_DIR.exists():
    sys.path.insert(0, str(PROPAINTER_DIR))


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
        self.propainter_available = self._check_propainter()
        
        print(f"[Inpainter] Device: {self.device}")
        print(f"[Inpainter] ProPainter path: {self.propainter_path}")
        print(f"[Inpainter] ProPainter available: {self.propainter_available}")
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
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
    
    def inpaint_video(
        self,
        video_path: str,
        masks: Dict[int, np.ndarray],
        output_path: str,
        method: str = 'auto',
        progress_callback: Callable = None
    ) -> str:
        """
        Inpaint video with given masks.
        
        Args:
            video_path: Input video path
            masks: Dict mapping frame_index -> binary mask (255 = inpaint)
            output_path: Output video path
            method: 'propainter', 'opencv', or 'auto'
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
            return self._inpaint_propainter(video_path, masks, output_path, progress_callback)
        else:
            return self._inpaint_opencv(video_path, masks, output_path, progress_callback)
    
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
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            mask = masks.get(frame_idx)
            
            if mask is not None and np.any(mask):
                inpainted = self._seamless_inpaint(frame, mask)
                out.write(inpainted)
            else:
                out.write(frame)
            
            if progress_callback:
                progress_callback(frame_idx + 1, total_frames, "Inpainting (OpenCV)")
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
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
        
        # Chunking parameters - tuned for RTX 4080 (16GB)
        CHUNK_SIZE = 40       # Frames per chunk
        OVERLAP = 10          # Overlap between chunks for blending
        
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
            print(f"[Inpainter] Chunk size: {CHUNK_SIZE}, Overlap: {OVERLAP}")
            
            # Calculate number of chunks
            if total_frames <= CHUNK_SIZE:
                num_chunks = 1
            else:
                effective_chunk = CHUNK_SIZE - OVERLAP
                num_chunks = max(1, (total_frames - OVERLAP + effective_chunk - 1) // effective_chunk)
            
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
                start_frame = chunk_idx * (CHUNK_SIZE - OVERLAP)
                end_frame = min(start_frame + CHUNK_SIZE, total_frames)
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
                self._run_propainter_chunk(
                    chunk_frames_dir, 
                    chunk_masks_dir, 
                    chunk_output_dir,
                    width, 
                    height,
                    len(chunk_frames)
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
                
                if result_video.exists():
                    # Extract frames from the result video
                    result_cap = cv2.VideoCapture(str(result_video))
                    result_frames = []
                    while True:
                        ret, frame = result_cap.read()
                        if not ret:
                            break
                        result_frames.append(frame)
                    result_cap.release()
                    
                    print(f"[Inpainter] Extracted {len(result_frames)} frames from ProPainter output")
                    
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
                                # This frame is in overlap zone - blend with previous chunk
                                overlap_position = local_idx  # How far into current chunk
                                blend_alpha = overlap_position / OVERLAP
                                
                                prev_frame = all_result_frames[global_idx]
                                blended = cv2.addWeighted(
                                    prev_frame, 1 - blend_alpha,
                                    result_frame, blend_alpha,
                                    0
                                )
                                all_result_frames[global_idx] = blended
                            else:
                                all_result_frames[global_idx] = result_frame
                else:
                    print(f"[Inpainter] ERROR: ProPainter output not found!")
                    print(f"[Inpainter] Searched: {result_video}")
                    # List contents for debugging
                    if chunk_output_dir.exists():
                        print(f"[Inpainter] Contents of {chunk_output_dir}:")
                        for item in chunk_output_dir.rglob("*"):
                            print(f"  - {item}")
                
                # Clean up chunk temp files
                shutil.rmtree(chunk_dir, ignore_errors=True)
            
            # Assemble final video
            if progress_callback:
                progress_callback(92, 100, "Assembling final video...")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for idx, frame in enumerate(all_result_frames):
                if frame is not None:
                    out.write(frame)
                else:
                    # Fallback to original if processing failed
                    out.write(all_frames[idx])
            
            out.release()
            
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
        num_frames: int
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
            "--resize_ratio", "1.0",
            "--height", str(height),
            "--width", str(width),
            # Temporal parameters for quality
            "--neighbor_length", "10",  # Full temporal window for quality
            "--ref_stride", "10",
            "--subvideo_length", str(num_frames),  # Process entire chunk as one
            "--save_fps", "30",
            "--fp16"  # FP16 saves memory without quality loss
        ]
        
        print(f"[Inpainter] Running ProPainter with memory optimizations...")
        print(f"[Inpainter] Command: {' '.join(cmd)}")
        
        # Set environment variable for better memory management
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
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
            if "OutOfMemoryError" in result.stderr or "out of memory" in result.stderr.lower():
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
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                out.write(frame)
        
        out.release()
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
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            mask = masks.get(frame_idx)
            if mask is not None and np.any(mask):
                frame = SimpleInpainter.inpaint_frame(frame, mask)
            
            out.write(frame)
            
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx + 1, total_frames, "Quick inpaint")
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        return output_path
