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
import socket
from pathlib import Path


def print_banner(port: int):
    """Print startup banner (ASCII-safe for Windows codepages)."""
    print(
        "\n"
        "==================================================================\n"
        "  WaterSlayer - Watermark Removal Suite\n"
        "  Version: 1.0.0\n"
        "------------------------------------------------------------------\n"
        f"  Open in browser:  http://localhost:{port}\n"
        "  Input videos:     ./data/input/\n"
        "  Output videos:    ./data/output/\n"
        "  Detection models: ./data/models/\n"
        "  Press Ctrl+C to stop\n"
        "==================================================================\n"
    )


def open_browser(url: str, delay: float = 1.5):
    """Open browser after a short delay."""

    def _open():
        time.sleep(delay)
        webbrowser.open(url)

    thread = threading.Thread(target=_open, daemon=True)
    thread.start()


def find_available_port(host: str, preferred_port: int, max_tries: int = 20) -> int:
    """Return first available port starting from preferred_port."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    for p in range(preferred_port, preferred_port + max_tries):
        with socket.socket(family, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise RuntimeError(
        f"No available ports in range {preferred_port}-{preferred_port + max_tries - 1}"
    )


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

    host_for_bind_check = args.host
    try:
        chosen_port = find_available_port(host_for_bind_check, args.port)
    except RuntimeError:
        # Fallback to the user requested port and let uvicorn report if bind fails.
        chosen_port = args.port

    if chosen_port != args.port:
        print(f"[Launcher] Port {args.port} is busy, using {chosen_port} instead.")

    print_banner(chosen_port)

    # Open browser
    if not args.no_browser:
        open_browser(f"http://localhost:{chosen_port}")

    # Start uvicorn
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host=args.host,
        port=chosen_port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
