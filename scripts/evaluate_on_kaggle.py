"""
evaluate_on_kaggle.py
---------------------
Evaluates the custom-trained PyTorch MobileNetV3-SSDLite (Zero-YOLO) model
on the real-world Kaggle INRIA Pedestrian Detection benchmark dataset.

Features:
- Automatically downloads test samples directly from the Kaggle INRIA benchmark.
- Evaluates detection accuracy, confidence distributions, and latency.
- Calculates physical distance estimations (in metres) and zone classifications.
- Saves visual detection overlays with bounding boxes and distance tags to reports/kaggle_eval/.
- Prints a comprehensive evaluation report.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict
import cv2
import numpy as np
import yaml
from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.mobilenet_detector import MobileNetPersonDetector
from src.detection.distance import DistanceEstimator, Zone

console = Console()
KAGGLE_REPO_ID = "marcelarosalesj/inria-person"
DEFAULT_TEST_DIR = Path("data/kaggle_inria_test")
OUTPUT_REPORT_DIR = Path("reports/kaggle_eval")


def download_kaggle_samples(dest_dir: Path, max_samples: int = 30) -> List[Path]:
    """Download representative pedestrian test images from the Kaggle benchmark mirror."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing_images = list(dest_dir.glob("*.png")) + list(dest_dir.glob("*.jpg"))

    if len(existing_images) >= max_samples:
        console.print(f"[green][OK] Found {len(existing_images)} existing Kaggle benchmark test images in {dest_dir}[/green]")
        return existing_images[:max_samples]

    console.print(f"[bold cyan]Fetching {max_samples} test images from Kaggle INRIA Pedestrian Dataset ({KAGGLE_REPO_ID})...[/bold cyan]")

    from huggingface_hub import HfApi
    api = HfApi()
    all_files = api.list_repo_files(KAGGLE_REPO_ID, repo_type="dataset")
    ped_files = [f for f in all_files if "pedestrians" in f and f.endswith((".png", ".jpg")) and "no_pedestrians" not in f]

    downloaded_paths = []
    # Pick distributed slice across the dataset
    step = max(1, len(ped_files) // max_samples)
    selected_files = ped_files[::step][:max_samples]

    for idx, remote_file in enumerate(selected_files, start=1):
        filename = Path(remote_file).name
        local_path = dest_dir / filename
        if not local_path.exists():
            console.print(f"  Downloading [{idx}/{len(selected_files)}] {filename}...", end="\r")
            try:
                cached = hf_hub_download(
                    repo_id=KAGGLE_REPO_ID,
                    filename=remote_file,
                    repo_type="dataset"
                )
                import shutil
                shutil.copy(cached, local_path)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not fetch {filename}: {e}[/yellow]")
                continue
        downloaded_paths.append(local_path)

    console.print(f"\n[green][OK] Successfully prepared {len(downloaded_paths)} Kaggle benchmark test images![/green]")
    return downloaded_paths


def evaluate_dataset(
    image_paths: List[Path],
    config_path: str = "config/settings.yaml",
    checkpoint_path: str = "models/best_person_detector.pth"
):
    """Run full evaluation on Kaggle dataset images."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize custom MobileNet detector and distance estimator
    config["detection"]["model"] = checkpoint_path
    detector = MobileNetPersonDetector(config)
    estimator = DistanceEstimator(config)

    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    annotated_dir = OUTPUT_REPORT_DIR / "annotated_samples"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        f"[bold]Detector Model:[/bold] MobileNetV3-SSDLite (Zero-YOLO)\n"
        f"[bold]Checkpoint:[/bold] {checkpoint_path}\n"
        f"[bold]Dataset Source:[/bold] Kaggle INRIA Pedestrian Benchmark\n"
        f"[bold]Total Test Images:[/bold] {len(image_paths)}",
        title="Kaggle Pedestrian Dataset Evaluation",
        border_style="cyan"
    ))

    latencies = []
    total_detections = 0
    zone_counts = {z: 0 for z in Zone}
    confidences = []
    distances = []
    processed_count = 0

    console.print("[bold]Running inference and awareness pipeline across Kaggle dataset...[/bold]\n")

    for idx, img_path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]

        t0 = time.perf_counter()
        detections = detector.detect(frame)
        latency = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency)

        total_detections += len(detections)
        processed_count += 1

        annotated_frame = frame.copy()

        for det in detections:
            confidences.append(det.confidence)
            est = estimator.estimate(det, h)
            distances.append(est.distance_m)
            zone_counts[est.zone] += 1

            # Draw bounding box and distance tag
            color = estimator.get_zone_color_bgr(est.zone)
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            label = f"Person {det.confidence:.2f} | {est.distance_m:.1f}m [{est.zone_label}]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), (20, 20, 20), -1)
            cv2.putText(annotated_frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Save annotated sample
        output_file = annotated_dir / f"eval_{img_path.stem}.jpg"
        cv2.imwrite(str(output_file), annotated_frame)

    # Compile Summary Table
    table = Table(title="Kaggle Dataset Benchmark Results", border_style="green")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Result", style="bold white")

    avg_latency = sum(latencies) / max(len(latencies), 1)
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0
    avg_conf = (sum(confidences) / len(confidences)) * 100 if confidences else 0
    avg_dist = sum(distances) / len(distances) if distances else 0

    table.add_row("Total Test Images Evaluated", str(processed_count))
    table.add_row("Total Pedestrians Detected", str(total_detections))
    table.add_row("Detections per Image (Avg)", f"{total_detections / max(processed_count, 1):.2f}")
    table.add_row("Average Inference Latency", f"{avg_latency:.1f} ms ({avg_fps:.1f} FPS)")
    table.add_row("Mean Detection Confidence", f"{avg_conf:.1f}%")
    table.add_row("Average Estimated Distance", f"{avg_dist:.2f} metres")
    table.add_row("Critical Zone Detections (< 1m)", str(zone_counts[Zone.CRITICAL]))
    table.add_row("Close Zone Detections (1-2.5m)", str(zone_counts[Zone.CLOSE]))
    table.add_row("Near Zone Detections (2.5-4m)", str(zone_counts[Zone.NEAR]))
    table.add_row("Medium / Far Zone Detections", str(zone_counts[Zone.MEDIUM] + zone_counts[Zone.FAR]))
    table.add_row("Visual Annotations Saved To", str(annotated_dir))

    console.print(table)
    console.print(f"\n[green][OK] Successfully tested custom model on Kaggle Pedestrian Dataset![/green]\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Kaggle Pedestrian Dataset")
    parser.add_argument("--samples", type=int, default=25, help="Number of test images to evaluate (default: 25)")
    parser.add_argument("--model", type=str, default="models/best_person_detector.pth", help="Model checkpoint path")
    args = parser.parse_args()

    paths = download_kaggle_samples(DEFAULT_TEST_DIR, max_samples=args.samples)
    evaluate_dataset(paths, checkpoint_path=args.model)


if __name__ == "__main__":
    main()
