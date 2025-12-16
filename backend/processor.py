"""
WaterSlayer - Video Processor

Main pipeline that combines detection and inpainting.
"""

import os
import cv2
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
    
    def load_model(self, model_path: str):
        """Load a YOLO detection model."""
        self.detector.load_model(model_path)
    
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
        inpaint_method: str = 'auto',
        progress_callback: Callable = None
    ) -> str:
        """
        Process a video to remove watermarks.
        
        Args:
            job: Processing job
            conf_threshold: Detection confidence threshold
            inpaint_method: 'propainter', 'opencv', or 'auto'
            progress_callback: Called with (progress, stage, message)
            
        Returns:
            Path to output video
        """
        try:
            # Load model if specified
            if job.model_path and job.model_path != self.detector.model_path:
                self.detector.load_model(job.model_path)
            
            if not self.detector.model:
                raise ValueError("No detection model loaded")
            
            # Open video
            cap = cv2.VideoCapture(job.input_path)
            job.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Stage 1: Detection
            job.status = JobStatus.DETECTING
            job.current_stage = "Detecting watermarks..."
            
            def detection_progress(current, total):
                job.processed_frames = current
                job.progress = (current / total) * 50  # 0-50%
                if progress_callback:
                    progress_callback(job.progress, "detecting", f"Frame {current}/{total}")
                
                if self._cancel_flags.get(job.job_id):
                    raise InterruptedError("Cancelled by user")
            
            # Detect watermarks
            detections_by_frame = self.detector.detect_video(
                job.input_path,
                conf_threshold=conf_threshold,
                sample_rate=1,  # Process every frame for best quality
                progress_callback=detection_progress
            )
            
            # Count total watermarks found
            job.watermarks_found = sum(len(dets) for dets in detections_by_frame.values())
            
            if job.watermarks_found == 0:
                # No watermarks found - just copy the video
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
                    mask = self.detector.create_mask(frame, dets, feather=3)
                    masks[frame_idx] = mask
                
                frame_idx += 1
            cap.release()
            
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
