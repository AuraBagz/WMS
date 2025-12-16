#!/usr/bin/env python3
"""
WaterSlayer - Unified Launcher

Starts both the FastAPI backend and serves the frontend.
Run with: python start.py

Options:
    --port PORT     Port to run on (default: 5346)
    --host HOST     Host to bind to (default: 0.0.0.0)
    --reload        Enable auto-reload for development
"""

import argparse
import webbrowser
import threading
import time
import sys
from pathlib import Path


def print_banner(port: int):
    """Print startup banner."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ⚔️  WaterSlayer - Watermark Removal Suite                      ║
║                                                                  ║
║   Version: 1.0.0                                                 ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   🌐 Open in browser:  http://localhost:{port:<5}                  ║
║                                                                  ║
║   📁 Input videos:     ./data/input/                             ║
║   📦 Output videos:    ./data/output/                            ║
║   🧠 Detection models: ./data/models/                            ║
║                                                                  ║
║   Press Ctrl+C to stop                                           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """.format(port=port))


def open_browser(url: str, delay: float = 1.5):
    """Open browser after a short delay."""
    def _open():
        time.sleep(delay)
        webbrowser.open(url)
    
    thread = threading.Thread(target=_open, daemon=True)
    thread.start()


def main():
    parser = argparse.ArgumentParser(
        description="WaterSlayer - Watermark Removal Suite"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5346,
        help="Port to run on (default: 5346)"
    )
    parser.add_argument(
        "--host", "-H",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--reload", "-r",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    
    args = parser.parse_args()
    
    # Create data directories
    data_dir = Path(__file__).parent / "data"
    (data_dir / "input").mkdir(parents=True, exist_ok=True)
    (data_dir / "output").mkdir(parents=True, exist_ok=True)
    (data_dir / "models").mkdir(parents=True, exist_ok=True)
    (data_dir / "temp").mkdir(parents=True, exist_ok=True)
    
    print_banner(args.port)
    
    # Open browser
    if not args.no_browser:
        open_browser(f"http://localhost:{args.port}")
    
    # Start uvicorn
    import uvicorn
    
    uvicorn.run(
        "backend.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
