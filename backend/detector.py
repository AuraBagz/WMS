"""
WaterSlayer - Watermark Detector

Uses YOLO models trained with AnnoStudio to detect watermarks.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO


class WatermarkDetector:
    """
    Detects watermarks in video frames using YOLO.
    
    Models are expected to be trained with AnnoStudio.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize detector.
        
        Args:
            model_path: Path to YOLO .pt weights file
        """
        self.model = None
        self.model_path = model_path
        self.device = 'cuda' if self._check_cuda() else 'cpu'
        
        if model_path:
            self.load_model(model_path)
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def load_model(self, model_path: str):
        """
        Load a YOLO model.
        
        Args:
            model_path: Path to .pt weights file
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = YOLO(str(path))
        self.model_path = str(path)
        print(f"[Detector] Loaded model: {path.name} (device: {self.device})")
    
    def detect(
        self, 
        frame: np.ndarray, 
        conf_threshold: float = 0.5,
        expand_ratio: float = 0.1
    ) -> List[Dict]:
        """
        Detect watermarks in a single frame.
        
        Args:
            frame: BGR image as numpy array
            conf_threshold: Minimum confidence threshold
            expand_ratio: Expand detected boxes by this ratio (for better inpainting)
            
        Returns:
            List of detection dicts with keys: x, y, width, height, confidence
        """
        if self.model is None:
            return []
        
        # Run inference
        results = self.model(frame, device=self.device, verbose=False)[0]
        
        detections = []
        h, w = frame.shape[:2]
        
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            
            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Convert to x, y, width, height
            box_w = x2 - x1
            box_h = y2 - y1
            
            # Expand box for better inpainting
            expand_w = box_w * expand_ratio
            expand_h = box_h * expand_ratio
            
            x1 = max(0, x1 - expand_w)
            y1 = max(0, y1 - expand_h)
            x2 = min(w, x2 + expand_w)
            y2 = min(h, y2 + expand_h)
            
            detections.append({
                'x': int(x1),
                'y': int(y1),
                'width': int(x2 - x1),
                'height': int(y2 - y1),
                'confidence': conf
            })
        
        return detections
    
    def create_mask(
        self, 
        frame: np.ndarray, 
        detections: List[Dict],
        feather: int = 15,
        use_ellipse: bool = True
    ) -> np.ndarray:
        """
        Create a smooth, feathered mask from detections.
        
        Args:
            frame: Original frame (for dimensions)
            detections: List of detection dicts
            feather: Feather radius for smooth edges (higher = softer blend)
            use_ellipse: Use ellipse instead of rectangle for more natural look
            
        Returns:
            Binary mask (255 = watermark region, 0 = keep)
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for det in detections:
            x, y = det['x'], det['y']
            bw, bh = det['width'], det['height']
            
            if use_ellipse:
                # Draw filled ellipse for more natural look
                center_x = x + bw // 2
                center_y = y + bh // 2
                axes = (bw // 2 + feather, bh // 2 + feather)
                cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)
            else:
                # Draw filled rectangle with rounded corners
                cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
        
        # Apply strong Gaussian blur for smooth feathered edges
        if feather > 0 and np.any(mask):
            # Use large kernel for smooth gradient
            blur_size = feather * 4 + 1
            if blur_size % 2 == 0:
                blur_size += 1
            
            # Create smooth gradient mask
            mask_float = mask.astype(np.float32) / 255.0
            mask_float = cv2.GaussianBlur(mask_float, (blur_size, blur_size), feather * 2)
            
            # Expand slightly to ensure full coverage
            mask_dilated = cv2.dilate(mask, np.ones((feather, feather), np.uint8), iterations=1)
            
            # Blend: use dilated for core, gradient for edges
            mask_float = cv2.GaussianBlur(mask_dilated.astype(np.float32) / 255.0, 
                                          (blur_size, blur_size), feather)
            
            # Threshold back to binary but with smoother edges
            mask = (mask_float * 255).astype(np.uint8)
            _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def create_soft_mask(
        self, 
        frame: np.ndarray, 
        detections: List[Dict],
        feather: int = 20
    ) -> np.ndarray:
        """
        Create a soft gradient mask for seamless blending.
        
        This mask has values from 0-255, not just binary, which allows
        for smoother inpainting transitions.
        
        Args:
            frame: Original frame (for dimensions)
            detections: List of detection dicts
            feather: Feather radius
            
        Returns:
            Gradient mask (0-255 values for soft blending)
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        for det in detections:
            x, y = det['x'], det['y']
            bw, bh = det['width'], det['height']
            
            # Create a smaller core region
            core_shrink = max(2, feather // 4)
            core_x = x + core_shrink
            core_y = y + core_shrink
            core_w = max(1, bw - core_shrink * 2)
            core_h = max(1, bh - core_shrink * 2)
            
            # Draw elliptical core at full intensity
            center_x = x + bw // 2
            center_y = y + bh // 2
            cv2.ellipse(mask, (center_x, center_y), 
                       (core_w // 2, core_h // 2), 0, 0, 360, 1.0, -1)
        
        # Heavy blur for soft gradient edges
        blur_size = feather * 6 + 1
        if blur_size % 2 == 0:
            blur_size += 1
        
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), feather * 3)
        
        # Normalize and convert
        if np.max(mask) > 0:
            mask = mask / np.max(mask)
        
        return (mask * 255).astype(np.uint8)
    
    def detect_video(
        self,
        video_path: str,
        conf_threshold: float = 0.5,
        sample_rate: int = 1,
        progress_callback=None
    ) -> Dict[int, List[Dict]]:
        """
        Detect watermarks in all frames of a video.
        
        Args:
            video_path: Path to video file
            conf_threshold: Confidence threshold
            sample_rate: Process every Nth frame (interpolate rest)
            progress_callback: Called with (current_frame, total_frames)
            
        Returns:
            Dict mapping frame_index -> list of detections
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        detections_by_frame = {}
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                detections = self.detect(frame, conf_threshold)
                detections_by_frame[frame_idx] = detections
            
            if progress_callback:
                progress_callback(frame_idx + 1, total_frames)
            
            frame_idx += 1
        
        cap.release()
        
        # Interpolate for skipped frames
        if sample_rate > 1:
            self._interpolate_detections(detections_by_frame, total_frames, sample_rate)
        
        return detections_by_frame
    
    def _interpolate_detections(
        self, 
        detections: Dict[int, List[Dict]], 
        total_frames: int,
        sample_rate: int
    ):
        """Fill in detections for skipped frames using nearest neighbor."""
        keyframes = sorted(detections.keys())
        
        for i in range(total_frames):
            if i in detections:
                continue
            
            # Find nearest keyframe
            nearest = min(keyframes, key=lambda k: abs(k - i))
            detections[i] = detections[nearest]
