"""
WaterSlayer Backend - FastAPI Application

REST API for watermark removal.
"""

import os
import asyncio
import threading
from pathlib import Path
from typing import List, Optional

import cv2

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from .processor import VideoProcessor, JobStatus
from .detector import WatermarkDetector


# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
MODELS_DIR = DATA_DIR / "models"
TEMP_DIR = DATA_DIR / "temp"

# Create directories
for d in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, MODELS_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Initialize processor
processor = VideoProcessor(OUTPUT_DIR)


# ============================================
# Pydantic Models
# ============================================

class ManualBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class ProcessRequest(BaseModel):
    video_path: str
    model_path: str = ''
    conf_threshold: float = 0.5
    high_threshold: Optional[float] = None
    low_threshold: Optional[float] = None
    detection_mode: str = 'standard'
    detail_restore_mode: str = 'off'
    inpaint_method: str = 'auto'
    quality_mode: str = 'auto'
    manual_box: Optional[ManualBox] = None


class ModelInfo(BaseModel):
    name: str
    filename: str
    path: str
    size: int


# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="WaterSlayer API",
    description="Watermark Removal Backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Video-Width", "X-Video-Height", "X-Total-Frames"],
)


# ============================================
# Video Endpoints
# ============================================

@app.get("/api/videos")
async def list_videos():
    """List all input videos."""
    videos = []
    
    for ext in ['*.mp4', '*.mov', '*.avi', '*.mkv', '*.webm']:
        for f in INPUT_DIR.glob(ext):
            videos.append({
                "name": f.stem,
                "filename": f.name,
                "path": str(f),
                "size": f.stat().st_size
            })
    
    # Sort by name
    videos.sort(key=lambda x: x["name"].lower())
    
    return {"videos": videos}


@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file."""
    # Save to input directory
    dest = INPUT_DIR / file.filename
    
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return {
        "success": True,
        "filename": file.filename,
        "path": str(dest),
        "size": dest.stat().st_size
    }


# ============================================
# Frame Preview Endpoint
# ============================================

@app.get("/api/videos/frame")
async def get_video_frame(path: str = Query(...), frame: int = Query(0)):
    """Extract a single frame from a video as JPEG for preview."""
    video_path = Path(path)
    if not video_path.exists():
        raise HTTPException(404, f"Video not found: {path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(400, "Could not open video")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not ret:
        raise HTTPException(400, "Could not read frame")

    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(
        content=buf.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Video-Width": str(w),
            "X-Video-Height": str(h),
            "X-Total-Frames": str(total)
        }
    )


# ============================================
# Model Endpoints
# ============================================

@app.get("/api/models")
async def list_models():
    """List all available detection models."""
    models = []
    
    for f in MODELS_DIR.glob("*.pt"):
        models.append({
            "name": f.stem,
            "filename": f.name,
            "path": str(f),
            "size": f.stat().st_size
        })
    
    # Sort by name
    models.sort(key=lambda x: x["name"].lower())
    
    return {"models": models}


@app.post("/api/models/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload a model file."""
    if not file.filename.endswith('.pt'):
        raise HTTPException(400, "Only .pt files are supported")
    
    dest = MODELS_DIR / file.filename
    
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return {
        "success": True,
        "filename": file.filename,
        "path": str(dest)
    }


@app.post("/api/models/import")
async def import_model(source_path: str = Query(...)):
    """Import a model from another location (e.g., AnnoStudio)."""
    source = Path(source_path)
    
    if not source.exists():
        raise HTTPException(404, f"Model not found: {source_path}")
    
    if not source.suffix == '.pt':
        raise HTTPException(400, "Only .pt files are supported")
    
    import shutil
    dest = MODELS_DIR / source.name
    shutil.copy(source, dest)
    
    return {
        "success": True,
        "filename": source.name,
        "path": str(dest)
    }


@app.post("/api/models/load")
async def load_model(model_path: str = Query(...)):
    """
    Pre-load a model into GPU memory for faster processing.
    
    The model stays cached until a different model is loaded.
    """
    path = Path(model_path)
    
    if not path.exists():
        raise HTTPException(404, f"Model not found: {model_path}")
    
    try:
        processor.load_model(str(path))
        
        return {
            "success": True,
            "model_path": str(path),
            "model_name": path.stem,
            "message": f"Model '{path.stem}' loaded successfully"
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {str(e)}")


@app.get("/api/models/loaded")
async def get_loaded_model():
    """Get the currently loaded model info."""
    if processor.detector.model is None:
        return {
            "loaded": False,
            "model_path": None,
            "model_name": None
        }
    
    model_path = processor.detector.model_path
    return {
        "loaded": True,
        "model_path": model_path,
        "model_name": Path(model_path).stem if model_path else None
    }


# ============================================
# Processing Endpoints
# ============================================

@app.post("/api/process")
async def start_processing(request: ProcessRequest):
    """Start watermark removal processing."""
    # Validate inputs
    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(404, f"Video not found: {request.video_path}")

    # Manual box mode: model is optional
    manual_box_dict = None
    if request.manual_box:
        manual_box_dict = {
            'x': request.manual_box.x,
            'y': request.manual_box.y,
            'width': request.manual_box.width,
            'height': request.manual_box.height,
        }
    else:
        model_path = Path(request.model_path)
        if not model_path.exists():
            raise HTTPException(404, f"Model not found: {request.model_path}")

    # Create job
    job = processor.create_job(str(video_path), request.model_path or "")

    # Start processing in background thread
    def run_processing():
        try:
            high_threshold = request.high_threshold if request.high_threshold is not None else request.conf_threshold
            low_threshold = request.low_threshold if request.low_threshold is not None else 0.2
            processor.process_video(
                job,
                conf_threshold=high_threshold,
                low_threshold=low_threshold,
                detection_mode=request.detection_mode,
                detail_restore_mode=request.detail_restore_mode,
                inpaint_method=request.inpaint_method,
                quality_mode=request.quality_mode,
                manual_box=manual_box_dict
            )
        except Exception as e:
            print(f"Processing failed: {e}")

    thread = threading.Thread(target=run_processing, daemon=True)
    thread.start()

    return {
        "success": True,
        "job_id": job.job_id,
        "message": "Processing started"
    }


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a processing job."""
    job = processor.get_job(job_id)
    
    if not job:
        raise HTTPException(404, "Job not found")
    
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a processing job."""
    job = processor.get_job(job_id)
    
    if not job:
        raise HTTPException(404, "Job not found")
    
    processor.cancel_job(job_id)
    
    return {"success": True, "message": "Cancellation requested"}


@app.get("/api/jobs")
async def list_jobs():
    """List all processing jobs."""
    return {"jobs": processor.list_jobs()}


# ============================================
# Output Endpoints
# ============================================

@app.get("/api/outputs")
async def list_outputs():
    """List all processed output videos."""
    outputs = []
    
    for f in OUTPUT_DIR.glob("*.mp4"):
        outputs.append({
            "name": f.stem,
            "filename": f.name,
            "path": str(f),
            "size": f.stat().st_size,
            "modified": f.stat().st_mtime
        })
    
    # Sort by modified time, newest first
    outputs.sort(key=lambda x: x["modified"], reverse=True)
    
    return {"outputs": outputs}


@app.get("/api/outputs/{filename}/download")
async def download_output(filename: str):
    """Download an output video."""
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(404, "Output not found")
    
    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=filename
    )


@app.delete("/api/outputs/{filename}")
async def delete_output(filename: str):
    """Delete an output video."""
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(404, "Output not found")
    
    file_path.unlink()
    
    return {"success": True, "message": f"Deleted {filename}"}


# ============================================
# System Endpoints
# ============================================

@app.get("/api/system/info")
async def get_system_info():
    """Get system information."""
    import torch
    
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pytorch_version": torch.__version__
    }


# ============================================
# Static Files & Frontend
# ============================================

# Serve frontend files
FRONTEND_DIR = BASE_DIR

@app.get("/")
async def serve_frontend():
    """Serve the main frontend HTML."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>WaterSlayer</h1><p>Frontend not found</p>")


@app.get("/{filename:path}")
async def serve_static(filename: str):
    """Serve static frontend files."""
    if filename.startswith("api/"):
        raise HTTPException(404)
    
    file_path = FRONTEND_DIR / filename
    
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    
    raise HTTPException(404, "File not found")
