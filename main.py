#!/usr/bin/env python3
"""
main.py
-------
CrowdAware AI — Entry Point

Usage examples:
    python main.py                         # Run with defaults
    python main.py --camera 1              # Use camera index 1
    python main.py --model yolov8s.pt      # Use larger model
    python main.py --no-audio              # Disable voice alerts
    python main.py --no-display            # Headless mode
    python main.py --config custom.yaml   # Use alternate config
    python main.py --source video.mp4      # Use video file instead of camera

Keyboard shortcuts (while running):
    Q / ESC   Quit
    C         Show calibration instructions
    S         Save screenshot
    A         Test audio alert

Author: CrowdAware AI Team
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from loguru import logger


console = Console()

# ------------------------------------------------------------------
# Banner
# ------------------------------------------------------------------

BANNER = """
 ██████╗██████╗  ██████╗ ██╗    ██╗██████╗      █████╗ ██╗    ██╗ █████╗ ██████╗ ███████╗
██╔════╝██╔══██╗██╔═══██╗██║    ██║██╔══██╗    ██╔══██╗██║    ██║██╔══██╗██╔══██╗██╔════╝
██║     ██████╔╝██║   ██║██║ █╗ ██║██║  ██║    ███████║██║ █╗ ██║███████║██████╔╝█████╗  
██║     ██╔══██╗██║   ██║██║███╗██║██║  ██║    ██╔══██║██║███╗██║██╔══██║██╔══██╗██╔══╝  
╚██████╗██║  ██║╚██████╔╝╚███╔███╔╝██████╔╝    ██║  ██║╚███╔███╔╝██║  ██║██║  ██║███████╗
 ╚═════╝╚═╝  ╚═╝ ╚═════╝  ╚══╝╚══╝ ╚═════╝     ╚═╝  ╚═╝ ╚══╝╚══╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
        AI-Based Person & Crowd Awareness System  |  v1.0.0
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CrowdAware AI — Real-time person detection, distance estimation & audio alerts",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Input source
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument(
        "--camera", "-c", type=int, default=None, metavar="INDEX",
        help="Camera device index (default: from config, usually 0)"
    )
    src_group.add_argument(
        "--source", "-s", type=str, default=None, metavar="PATH",
        help="Path to video file (MP4, AVI, etc.) instead of live camera"
    )

    # Model
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="YOLOv8 model size (n=fastest, x=most accurate)"
    )

    # Config
    parser.add_argument(
        "--config", type=str, default="config/settings.yaml",
        help="Path to YAML configuration file (default: config/settings.yaml)"
    )

    # Overrides
    parser.add_argument("--no-audio",   action="store_true", help="Disable audio alerts")
    parser.add_argument("--no-display", action="store_true", help="Headless (no video window)")
    parser.add_argument("--no-track",   action="store_true", help="Disable person tracking")

    parser.add_argument(
        "--confidence", type=float, default=None, metavar="FLOAT",
        help="Override detection confidence threshold (0.0–1.0)"
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda", "mps"],
        help="Force compute device (overrides auto-detection)"
    )

    # Debug
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity"
    )
    parser.add_argument("--version", action="version", version="CrowdAware AI v1.0.0")

    return parser.parse_args()


# ------------------------------------------------------------------
# Config loader
# ------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and return the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        console.print("[yellow]Using built-in defaults.[/yellow]")
        return {}
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return cfg or {}


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """
    Merge CLI arguments into the loaded config.
    CLI flags take priority over config file values.
    """
    if args.camera is not None:
        config.setdefault("camera", {})["device_id"] = args.camera

    if args.source is not None:
        config.setdefault("camera", {})["device_id"] = args.source

    if args.model is not None:
        config.setdefault("detection", {})["model"] = args.model

    if args.confidence is not None:
        config.setdefault("detection", {})["confidence_threshold"] = args.confidence

    if args.device is not None:
        config.setdefault("detection", {})["device"] = args.device

    if args.no_audio:
        config.setdefault("audio", {})["enabled"] = False

    if args.no_display:
        config.setdefault("display", {})["enabled"] = False

    if args.no_track:
        config.setdefault("tracking", {})["enabled"] = False

    return config


# ------------------------------------------------------------------
# Setup logging
# ------------------------------------------------------------------

def setup_logging(level: str, config: dict):
    """Configure loguru logging."""
    log_cfg = config.get("logging", {})
    log_file = log_cfg.get("file", "logs/crowdaware.log")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Remove default loguru handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )

    # File handler
    logger.add(
        log_file,
        level="DEBUG",
        rotation=f"{log_cfg.get('max_file_size_mb', 10)} MB",
        retention=log_cfg.get("backup_count", 3),
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
    )


# ------------------------------------------------------------------
# Pre-flight checks
# ------------------------------------------------------------------

def run_preflight_checks(config: dict) -> bool:
    """Verify required dependencies and camera availability."""
    console.print("\n[bold]Running pre-flight checks…[/bold]")
    ok = True

    # Python version
    if sys.version_info < (3, 8):
        console.print(f"[red]✗ Python 3.8+ required (found {sys.version})[/red]")
        ok = False
    else:
        console.print(f"[green]✓ Python {sys.version_info.major}.{sys.version_info.minor}[/green]")

    # OpenCV
    try:
        import cv2
        console.print(f"[green]✓ OpenCV {cv2.__version__}[/green]")
    except ImportError:
        console.print("[red]✗ OpenCV not installed (pip install opencv-python)[/red]")
        ok = False

    # PyTorch
    try:
        import torch
        device_info = "CPU"
        if torch.cuda.is_available():
            device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_info = "Apple MPS"
        console.print(f"[green]✓ PyTorch {torch.__version__} — {device_info}[/green]")
    except ImportError:
        console.print("[red]✗ PyTorch not installed (pip install torch)[/red]")
        ok = False

    # Ultralytics (YOLOv8)
    try:
        import ultralytics
        console.print(f"[green]✓ Ultralytics YOLOv8 {ultralytics.__version__}[/green]")
    except ImportError:
        console.print("[red]✗ Ultralytics not installed (pip install ultralytics)[/red]")
        ok = False

    # Audio
    audio_enabled = config.get("audio", {}).get("enabled", True)
    if audio_enabled:
        try:
            import pyttsx3
            console.print("[green]✓ pyttsx3 TTS available[/green]")
        except ImportError:
            console.print("[yellow]⚠ pyttsx3 not installed — alerts will be text only[/yellow]")

    console.print()
    return ok


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    # Print banner
    console.print(f"[bold cyan]{BANNER}[/bold cyan]")

    # Load config
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    # Logging
    setup_logging(args.log_level, config)

    # Print config summary
    det_model = config.get("detection", {}).get("model", "yolov8n.pt")
    cam_id    = config.get("camera", {}).get("device_id", 0)
    audio_on  = config.get("audio", {}).get("enabled", True)

    console.print(Panel(
        f"[bold]Model:[/bold] {det_model}\n"
        f"[bold]Camera:[/bold] {cam_id}\n"
        f"[bold]Audio:[/bold] {'enabled' if audio_on else 'disabled'}\n"
        f"[bold]Config:[/bold] {args.config}",
        title="CrowdAware AI Configuration",
        border_style="cyan",
    ))

    # Pre-flight
    if not run_preflight_checks(config):
        console.print("[red]Pre-flight checks failed. Please install missing dependencies.[/red]")
        console.print("[yellow]Run:  pip install -r requirements.txt[/yellow]")
        sys.exit(1)

    # Launch pipeline
    from src.pipeline import CrowdAwarePipeline
    pipeline = CrowdAwarePipeline(config)

    console.print("[bold green]Starting CrowdAware AI…[/bold green]")
    console.print("[dim]Press Q or ESC in the video window to quit.[/dim]\n")

    pipeline.run()


if __name__ == "__main__":
    main()
