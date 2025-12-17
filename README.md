# ⚔️ WaterSlayer

**AI-Powered Video Watermark Removal Suite**

A standalone, portable application that removes watermarks from videos using YOLO detection and ProPainter inpainting.

---

## ✨ Features

- **🎯 YOLO Detection** - Uses models trained with AnnoStudio
- **🎨 ProPainter Inpainting** - State-of-the-art video inpainting
- **🔊 Audio Preservation** - Keeps original audio in output videos
- **⚡ GPU Accelerated** - Optimized for NVIDIA RTX GPUs
- **📦 Portable** - Standalone app, easy to install
- **🌐 Web UI** - Modern, clean interface

---

## 🚀 Quick Start

### 1. Install

```batch
INSTALL.bat
```

This will:
- Create a virtual environment
- Install PyTorch with CUDA 12.4
- Install all dependencies
- Install FFmpeg for audio preservation

### 2. Import a Detection Model

Copy your trained model from AnnoStudio:
```
From: AnnoStudio/data/weights/your_model.pt
To:   WaterSlayer/data/models/your_model.pt
```

Or use the "Import" button in the UI.

### 3. Add Input Videos

Place videos in:
```
WaterSlayer/data/input/
```

### 4. Run

```batch
START.bat
```

Opens in browser at `http://localhost:5346`

---

## 📁 Structure

```
WaterSlayer/
├── INSTALL.bat           # One-click installer
├── START.bat             # Launch app
├── start.py              # Python launcher
├── requirements.txt      # Dependencies
├── index.html            # Frontend UI
├── styles.css            # Styling
├── app.js                # Frontend logic
├── backend/
│   ├── app.py            # FastAPI server
│   ├── detector.py       # YOLO detection
│   ├── inpainter.py      # Video inpainting
│   └── processor.py      # Pipeline
└── data/
    ├── input/            # Input videos
    ├── output/           # Processed videos
    ├── models/           # Detection models (.pt)
    └── temp/             # Temporary files
```

---

## 🔧 Processing Pipeline

1. **Load Video** - Read input video frames
2. **Detect Watermarks** - YOLO model finds watermark regions
3. **Create Masks** - Generate binary masks from detections
4. **Inpaint** - Remove watermarks using ProPainter or OpenCV
5. **Mux Audio** - Add original audio back using FFmpeg
6. **Export** - Save clean video with audio

---

## ⚙️ Settings

| Setting | Description |
|---------|-------------|
| **Detection Confidence** | Higher = fewer false positives, but might miss subtle watermarks |
| **Inpainting Method** | `auto` uses best available, `opencv` is fast, `propainter` is high-quality |

---

## 🔗 Integration with AnnoStudio

1. **Train a model** in AnnoStudio
2. **Download** the trained `.pt` file
3. **Import** into WaterSlayer
4. **Process** videos!

---

## 🖥️ System Requirements

- **OS**: Windows 10/11
- **Python**: 3.10+
- **GPU**: NVIDIA RTX (recommended for fast processing)
- **VRAM**: 4GB+ recommended
- **FFmpeg**: Required for audio preservation (auto-installed)

---

## 📜 License

MIT License

---

<p align="center">
  Made with ⚔️ by the WaterSlayer Team
</p>
