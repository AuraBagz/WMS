"""
WaterSlayer - Video Processor

Main pipeline that combines detection and inpainting.
"""

import os
import cv2
import numpy as np
import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

from .detector import WatermarkDetector
from .inpainter import VideoInpainter, SimpleInpainter


class JobStatus(Enum):
    PENDING = "pending"
    DETECTING = "detecting"
    INPAINTING = "inpainting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingJob:
    """Represents a video processing job."""
    job_id: str
    input_path: str
    output_path: str = ""
    model_path: str = ""
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    current_stage: str = ""
    error: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = ""
    
    # Processing stats
    total_frames: int = 0
    processed_frames: int = 0
    watermarks_found: int = 0
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['status'] = self.status.value
        return d


class VideoProcessor:
    """
    Main video processing pipeline.
    
    1. Load video
    2. Detect watermarks in all frames
    3. Create masks from detections
    4. Inpaint masked regions
    5. Save clean video
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize processor.
        
        Args:
            output_dir: Directory for output videos
        """
        self.output_dir = output_dir
        self.detector = WatermarkDetector()
        self.inpainter = VideoInpainter()
        
        self.jobs: Dict[str, ProcessingJob] = {}
        self._cancel_flags: Dict[str, bool] = {}
        self._loaded_model_sig: Optional[tuple] = None
    
    def load_model(self, model_path: str):
        """Load a YOLO detection model."""
        self.detector.load_model(model_path)
        p = Path(model_path)
        try:
            st = p.stat()
            self._loaded_model_sig = (str(p.resolve()), st.st_mtime_ns, st.st_size)
        except Exception:
            self._loaded_model_sig = (str(p), None, None)

    def _model_sig(self, model_path: str) -> tuple:
        p = Path(model_path)
        st = p.stat()
        return (str(p.resolve()), st.st_mtime_ns, st.st_size)

    def _stabilize_masks_temporal(
        self,
        masks: Dict[int, np.ndarray],
        total_frames: int,
        frame_height: int,
        frame_width: int,
        window: int = 5,
        persist: float = 0.35
    ) -> Dict[int, np.ndarray]:
        """
        Stabilize binary masks over time to reduce popping/flicker.

        Technique:
        1) Median over a short temporal window (3-5 frames).
        2) Decayed union with previous stabilized mask.
        """
        if not masks or total_frames <= 1:
            return masks

        window = max(3, int(window))
        if window % 2 == 0:
            window += 1

        half = window // 2
        prev_prob = np.zeros((frame_height, frame_width), dtype=np.float32)
        stabilized: Dict[int, np.ndarray] = {}

        for frame_idx in range(total_frames):
            start = max(0, frame_idx - half)
            end = min(total_frames, frame_idx + half + 1)

            stack = []
            for t in range(start, end):
                m = masks.get(t)
                if m is None:
                    stack.append(np.zeros((frame_height, frame_width), dtype=np.uint8))
                else:
                    if m.shape[:2] != (frame_height, frame_width):
                        m = cv2.resize(m, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                    stack.append((m > 127).astype(np.uint8))

            win = np.stack(stack, axis=0)
            votes = np.sum(win, axis=0)
            med = (votes >= ((win.shape[0] // 2) + 1)).astype(np.float32)

            prev_prob *= float(persist)
            prev_prob = np.maximum(prev_prob, med)
            stable = (prev_prob >= 0.5)

            if np.any(stable):
                stabilized[frame_idx] = (stable.astype(np.uint8) * 255)

        return stabilized
    
    def create_job(self, input_path: str, model_path: str = None) -> ProcessingJob:
        """
        Create a new processing job.
        
        Args:
            input_path: Path to input video
            model_path: Path to YOLO model (optional if already loaded)
            
        Returns:
            ProcessingJob instance
        """
        job_id = str(uuid.uuid4())[:8]
        
        # Generate output filename
        input_name = Path(input_path).stem
        output_name = f"{input_name}_clean_{job_id}.mp4"
        output_path = str(self.output_dir / output_name)
        
        job = ProcessingJob(
            job_id=job_id,
            input_path=input_path,
            output_path=output_path,
            model_path=model_path or self.detector.model_path or ""
        )
        
        self.jobs[job_id] = job
        self._cancel_flags[job_id] = False
        
        return job
    
    def process_video(
        self,
        job: ProcessingJob,
        conf_threshold: float = 0.5,
        low_threshold: float = 0.2,
        detection_mode: str = 'standard',
        detail_restore_mode: str = 'off',
        inpaint_method: str = 'auto',
        quality_mode: str = 'balanced',
        progress_callback: Callable = None,
        manual_box: Optional[Dict] = None
    ) -> str:
        """
        Process a video to remove watermarks.
        
        Args:
            job: Processing job
            conf_threshold: High detection confidence threshold
            low_threshold: Low detection confidence threshold used for mask continuity
            detection_mode: 'standard' or 'strict_parity'
            detail_restore_mode: 'off', 'roi_sharpen', 'roi_sharpen_strong'
            inpaint_method: 'propainter', 'opencv', or 'auto'
            quality_mode: 'balanced', 'rtx5090', 'rtx5090_crisp', or 'auto'
            progress_callback: Called with (progress, stage, message)
            
        Returns:
            Path to output video
        """
        try:
            # Open video
            cap = cv2.VideoCapture(job.input_path)
            job.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Resolve quality mode
            effective_quality_mode = quality_mode
            if quality_mode == 'auto':
                try:
                    import torch
                    gpu_name = torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else ""
                    effective_quality_mode = 'rtx5090' if '5090' in gpu_name else 'balanced'
                except Exception:
                    effective_quality_mode = 'balanced'

            if manual_box:
                # ── Manual Box Mode: skip YOLO, create static mask ──
                job.status = JobStatus.DETECTING
                job.current_stage = "Creating mask from manual box..."
                job.progress = 25

                manual_det = [{
                    'x': manual_box['x'],
                    'y': manual_box['y'],
                    'width': manual_box['width'],
                    'height': manual_box['height'],
                    'confidence': 1.0
                }]

                # Create one mask and apply to every frame
                dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
                feather = 1 if effective_quality_mode == 'rtx5090_crisp' else 3
                single_mask = self.detector.create_mask(dummy_frame, manual_det, feather=feather)

                masks = {}
                for i in range(job.total_frames):
                    masks[i] = single_mask

                job.watermarks_found = job.total_frames
                job.progress = 50

            else:
                # ── YOLO Detection Mode ──
                # Load model if needed
                if job.model_path:
                    try:
                        desired_sig = self._model_sig(job.model_path)
                    except Exception:
                        desired_sig = (str(Path(job.model_path)), None, None)

                    current_path = str(Path(self.detector.model_path).resolve()) if self.detector.model_path else None
                    if (
                        self.detector.model is None
                        or current_path != desired_sig[0]
                        or self._loaded_model_sig != desired_sig
                    ):
                        self.load_model(job.model_path)

                if not self.detector.model:
                    raise ValueError("No detection model loaded")

                # Stage 1: Detection
                job.status = JobStatus.DETECTING
                job.current_stage = "Detecting watermarks..."

                def detection_progress(current, total):
                    job.processed_frames = current
                    job.progress = (current / total) * 50
                    if progress_callback:
                        progress_callback(
                            job.progress,
                            "detecting",
                            f"Frame {current}/{total} (high={conf_threshold:.2f}, low={low_threshold:.2f}, mode={detection_mode})"
                        )
                    if self._cancel_flags.get(job.job_id):
                        raise InterruptedError("Cancelled by user")

                effective_low = min(float(conf_threshold), max(0.01, float(low_threshold)))
                box_expand_ratio = 0.04 if effective_quality_mode == 'rtx5090_crisp' else 0.10
                detections_by_frame = self.detector.detect_video(
                    job.input_path,
                    conf_threshold=effective_low,
                    expand_ratio=box_expand_ratio,
                    sample_rate=1,
                    progress_callback=detection_progress
                )

                if detection_mode != 'strict_parity':
                    if effective_quality_mode == 'rtx5090':
                        detections_by_frame = self.detector.smooth_video_detections(
                            detections_by_frame, total_frames=job.total_frames,
                            alpha=0.58, max_shift_ratio=0.25, hold_frames=6
                        )
                    elif effective_quality_mode == 'rtx5090_crisp':
                        detections_by_frame = self.detector.smooth_video_detections(
                            detections_by_frame, total_frames=job.total_frames,
                            alpha=0.70, max_shift_ratio=0.20, hold_frames=3
                        )
                    else:
                        detections_by_frame = self.detector.smooth_video_detections(
                            detections_by_frame, total_frames=job.total_frames,
                            alpha=0.72, max_shift_ratio=0.35, hold_frames=4
                        )

                job.watermarks_found = sum(len(dets) for dets in detections_by_frame.values())

                if job.watermarks_found == 0:
                    job.status = JobStatus.COMPLETED
                    job.progress = 100
                    job.completed_at = datetime.now().isoformat()
                    import shutil
                    shutil.copy(job.input_path, job.output_path)
                    return job.output_path

                # Create masks from detections
                masks = {}
                cap = cv2.VideoCapture(job.input_path)
                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    dets = detections_by_frame.get(frame_idx, [])
                    if dets:
                        feather = 1 if effective_quality_mode == 'rtx5090_crisp' else 3
                        mask = self.detector.create_mask(frame, dets, feather=feather)
                        masks[frame_idx] = mask
                    frame_idx += 1
                cap.release()

                # Extra temporal mask stabilization for 5090 quality profiles
                if effective_quality_mode in ('rtx5090', 'rtx5090_crisp'):
                    masks = self._stabilize_masks_temporal(
                        masks, total_frames=job.total_frames,
                        frame_height=height, frame_width=width,
                        window=5, persist=0.35
                    )
            
            # Stage 2: Inpainting
            job.status = JobStatus.INPAINTING
            job.current_stage = "Removing watermarks..."
            
            def inpaint_progress(current, total, message):
                job.progress = 50 + (current / total) * 50  # 50-100%
                if progress_callback:
                    progress_callback(job.progress, "inpainting", message)
                
                if self._cancel_flags.get(job.job_id):
                    raise InterruptedError("Cancelled by user")
            
            # Inpaint
            output_path = self.inpainter.inpaint_video(
                job.input_path,
                masks,
                job.output_path,
                method=inpaint_method,
                quality_mode=quality_mode,
                detail_restore_mode=detail_restore_mode,
                progress_callback=inpaint_progress
            )
            
            # Complete
            job.status = JobStatus.COMPLETED
            job.progress = 100
            job.completed_at = datetime.now().isoformat()
            job.output_path = output_path
            
            return output_path
            
        except InterruptedError:
            job.status = JobStatus.CANCELLED
            job.error = "Cancelled by user"
            raise
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            raise
    
    def cancel_job(self, job_id: str):
        """Cancel a running job."""
        self._cancel_flags[job_id] = True
    
    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get a job by ID."""
        return self.jobs.get(job_id)
    
    def list_jobs(self) -> list:
        """List all jobs."""
        return [job.to_dict() for job in self.jobs.values()]


def quick_remove_watermark(
    video_path: str,
    model_path: str,
    output_path: str,
    conf_threshold: float = 0.5
) -> str:
    """
    Quick watermark removal for command-line usage.
    
    Args:
        video_path: Input video
        model_path: YOLO model path
        output_path: Output video path
        conf_threshold: Detection threshold
        
    Returns:
        Path to clean video
    """
    detector = WatermarkDetector(model_path)
    
    # Detect
    print("Detecting watermarks...")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    detections = detector.detect_video(
        video_path,
        conf_threshold=conf_threshold,
        progress_callback=lambda c, t: print(f"\rFrame {c}/{t}", end="")
    )
    print()
    
    watermark_count = sum(len(d) for d in detections.values())
    print(f"Found {watermark_count} watermark instances")
    
    if watermark_count == 0:
        print("No watermarks found!")
        return video_path
    
    # Create masks
    print("Creating masks...")
    masks = {}
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        dets = detections.get(idx, [])
        if dets:
            masks[idx] = detector.create_mask(frame, dets)
        idx += 1
    cap.release()
    
    # Inpaint
    print("Inpainting...")
    SimpleInpainter.inpaint_video_simple(
        video_path,
        masks,
        output_path,
        progress_callback=lambda c, t, m: print(f"\rFrame {c}/{t}", end="")
    )
    print()
    
    print(f"Done! Output: {output_path}")
    return output_path
