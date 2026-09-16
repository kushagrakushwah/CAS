#!/usr/bin/env python3
"""
benchmark.py
------------
Performance benchmarking tool for CrowdAware AI.

Measures:
- Detection inference speed (FPS & ms per frame)
- Distance estimation throughput
- Tracking throughput
- Full pipeline FPS

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --model yolov8m.pt --frames 200
    python scripts/benchmark.py --device cpu --device cuda   # compare

Author: CrowdAware AI Team
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="CrowdAware AI Benchmark")
    p.add_argument("--frames",  type=int,   default=100, help="Frames to benchmark")
    p.add_argument("--model",   type=str,   default="yolov8n.pt")
    p.add_argument("--device",  type=str,   default="auto")
    p.add_argument("--width",   type=int,   default=1280)
    p.add_argument("--height",  type=int,   default=720)
    p.add_argument("--warmup",  type=int,   default=10, help="Warm-up frames")
    return p.parse_args()


def make_test_frames(n, w, h):
    """Generate N random test frames."""
    return [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def benchmark_detector(config, frames, warmup):
    from src.detection.detector import PersonDetector
    det = PersonDetector(config)

    print(f"\n  Warming up ({warmup} frames)…")
    for f in frames[:warmup]:
        det.detect(f)

    print(f"  Benchmarking ({len(frames)} frames)…")
    t0 = time.perf_counter()
    total_dets = 0
    for f in frames:
        dets = det.detect(f)
        total_dets += len(dets)
    elapsed = time.perf_counter() - t0

    return {
        "total_frames": len(frames),
        "elapsed_s": elapsed,
        "fps": len(frames) / elapsed,
        "ms_per_frame": elapsed * 1000 / len(frames),
        "avg_detections": total_dets / len(frames),
        "model": config["detection"]["model"],
        "device": det.device,
    }


def benchmark_tracker(config, n_detections, n_frames):
    from src.detection.detector import Detection
    from src.detection.tracker import MultiPersonTracker

    tracker = MultiPersonTracker(config)

    # Generate fake detection sequences
    dets = [
        Detection(
            bbox=[100 + i*120, 100, 200 + i*120, 450],
            confidence=0.9,
            class_id=0,
            class_name="person",
        )
        for i in range(n_detections)
    ]

    t0 = time.perf_counter()
    for _ in range(n_frames):
        tracker.update(dets, [None] * n_detections, (720, 1280))
    elapsed = time.perf_counter() - t0

    return {
        "n_frames": n_frames,
        "n_detections_per_frame": n_detections,
        "fps": n_frames / elapsed,
        "ms_per_frame": elapsed * 1000 / n_frames,
    }


def print_results(title, results):
    print(f"\n  {'─'*40}")
    print(f"  {title}")
    print(f"  {'─'*40}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<28}: {v:>10.2f}")
        else:
            print(f"  {k:<28}: {v!s:>10}")


def main():
    args = parse_args()

    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)

    config["detection"]["model"]  = args.model
    config["detection"]["device"] = args.device
    config["audio"]["enabled"]    = False

    print(f"\n{'='*50}")
    print(f"  CrowdAware AI — Performance Benchmark")
    print(f"{'='*50}")
    print(f"  Model  : {args.model}")
    print(f"  Device : {args.device}")
    print(f"  Frames : {args.frames}")
    print(f"  Size   : {args.width}×{args.height}")

    frames = make_test_frames(args.frames, args.width, args.height)

    # 1. Detector
    print("\n[1/3] Detection benchmark…")
    det_results = benchmark_detector(config, frames, args.warmup)
    print_results("DETECTION RESULTS", det_results)

    # 2. Tracker (with simulated detections)
    print("\n[2/3] Tracker benchmark (5 persons)…")
    trk_results = benchmark_tracker(config, n_detections=5, n_frames=args.frames)
    print_results("TRACKER RESULTS", trk_results)

    # 3. Full pipeline (no camera, no display)
    print("\n[3/3] Estimator throughput…")
    from src.detection.detector import Detection
    from src.detection.distance import DistanceEstimator

    estimator = DistanceEstimator(config)
    fake_dets = [
        Detection(bbox=[100, 50, 250, 450], confidence=0.9, class_id=0, class_name="person")
    ] * 5

    t0 = time.perf_counter()
    for _ in range(args.frames):
        for d in fake_dets:
            estimator.estimate(d, args.height)
    elapsed = time.perf_counter() - t0

    est_results = {
        "total_estimates": args.frames * len(fake_dets),
        "elapsed_s": elapsed,
        "estimates_per_second": (args.frames * len(fake_dets)) / elapsed,
    }
    print_results("ESTIMATOR RESULTS", est_results)

    # Summary
    print(f"\n{'='*50}")
    print(f"  SUMMARY")
    print(f"{'='*50}")
    det_fps = det_results['fps']
    if det_fps >= 30:
        grade = "✅ EXCELLENT (real-time)"
    elif det_fps >= 15:
        grade = "✅ GOOD (smooth)"
    elif det_fps >= 10:
        grade = "⚠️  ACCEPTABLE (may stutter)"
    else:
        grade = "❌ SLOW (use smaller model or GPU)"

    print(f"  Detection FPS : {det_fps:.1f}  →  {grade}")
    print(f"  Tracker FPS   : {trk_results['fps']:.1f}")
    print()
    if det_fps < 15:
        print("  💡 Tip: Try yolov8n.pt (smallest model) or add a GPU.")
    print()


if __name__ == "__main__":
    main()
