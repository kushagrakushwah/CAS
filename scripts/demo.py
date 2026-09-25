#!/usr/bin/env python3
"""
demo.py
-------
Demo mode — runs the CrowdAware system on synthetic or recorded video.

Useful for:
- Testing without a live camera
- CI/CD smoke tests
- Demonstrating the system to stakeholders

Usage:
    # Synthetic animated demo (no camera needed)
    python scripts/demo.py --synthetic

    # Run on a video file
    python scripts/demo.py --video path/to/video.mp4

    # Run on a directory of images
    python scripts/demo.py --images path/to/frames/

Author: CrowdAware AI Team
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="CrowdAware AI Demo")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--synthetic", action="store_true", help="Generate animated synthetic demo")
    grp.add_argument("--video",  type=str, help="Path to video file")
    grp.add_argument("--images", type=str, help="Directory of image files")
    p.add_argument("--frames", type=int, default=300, help="Number of frames for synthetic demo")
    p.add_argument("--fps",    type=int, default=30,  help="Demo FPS")
    p.add_argument("--save",   type=str, default=None, help="Save output to video file")
    return p.parse_args()


# ------------------------------------------------------------------
# Synthetic scene generator
# ------------------------------------------------------------------

class SyntheticSceneGenerator:
    """
    Generates a realistic synthetic crowd scene for testing.

    Draws animated 'persons' (colored rectangles that move around
    a scene) to simulate real-world camera input.
    """

    def __init__(self, width=1280, height=720):
        self.w = width
        self.h = height
        self.bg_color = (30, 35, 45)  # Dark indoor

        # Animated persons: [cx, cy, vx, vy, w, h, color]
        self.persons = [
            {"cx": 300, "cy": 400, "vx": 1.2, "vy": 0.0,  "w": 80, "h": 200, "color": (180, 120, 80)},
            {"cx": 900, "cy": 380, "vx": -0.8,"vy": 0.3,  "w": 75, "h": 190, "color": (100, 160, 200)},
            {"cx": 640, "cy": 200, "vx": 0.0, "vy": 1.5,  "w": 90, "h": 210, "color": (200, 180, 100)},
            {"cx": 150, "cy": 600, "vx": 2.5, "vy": -0.5, "w": 70, "h": 185, "color": (150, 200, 150)},
            {"cx": 1100,"cy": 500, "vx": -1.5,"vy": -0.8, "w": 85, "h": 195, "color": (200, 100, 150)},
        ]

    def generate_frame(self, frame_num: int) -> np.ndarray:
        """Generate a synthetic frame with animated persons."""
        frame = np.full((self.h, self.w, 3), self.bg_color, dtype=np.uint8)

        # Draw floor grid
        for x in range(0, self.w, 80):
            cv2.line(frame, (x, self.h//2), (x, self.h), (50, 50, 60), 1)
        cv2.line(frame, (0, self.h//2), (self.w, self.h//2), (60, 60, 70), 1)

        # Simulate distance fog on far objects
        # Update and draw each person
        for p in self.persons:
            # Update position
            p["cx"] = (p["cx"] + p["vx"]) % self.w
            p["cy"] = p["cy"] + p["vy"] * math.sin(frame_num * 0.02)

            # Keep in vertical bounds with bounce
            if p["cy"] < self.h // 3:
                p["cy"] = self.h // 3
            if p["cy"] > self.h - p["h"] // 2:
                p["cy"] = self.h - p["h"] // 2

            cx, cy = int(p["cx"]), int(p["cy"])
            w2 = p["w"] // 2
            h2 = p["h"] // 2

            x1, y1 = cx - w2, cy - h2
            x2, y2 = cx + w2, cy + h2

            # Body
            cv2.rectangle(frame, (x1, y1), (x2, y2), p["color"], -1)
            # Head
            head_r = p["w"] // 4
            cv2.circle(frame, (cx, y1 - head_r), head_r, p["color"], -1)

            # Shadow
            shadow_pts = np.array([
                [x1 + 10, y2],
                [x2 - 10, y2],
                [x2 + 5,  y2 + 10],
                [x1 - 5,  y2 + 10],
            ], np.int32)
            shadow_overlay = frame.copy()
            cv2.fillPoly(shadow_overlay, [shadow_pts], (20, 20, 25))
            cv2.addWeighted(shadow_overlay, 0.5, frame, 0.5, 0, frame)

        # Timestamp watermark
        t = time.strftime("%H:%M:%S")
        cv2.putText(frame, f"SYNTHETIC DEMO | {t}", (10, self.h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

        return frame


# ------------------------------------------------------------------
# Main demo runners
# ------------------------------------------------------------------

def run_synthetic_demo(args):
    print("\n[Demo] Synthetic mode — generating animated crowd scene…")
    print("[Demo] Press Q or ESC to quit.\n")

    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)

    # Disable audio for demo
    config.setdefault("audio", {})["enabled"] = False
    # Disable window (we'll handle it here)
    config.setdefault("display", {})["enabled"] = False

    from src.detection.detector import PersonDetector
    from src.detection.mobilenet_detector import MobileNetPersonDetector
    from src.detection.distance import DistanceEstimator
    from src.detection.tracker import MultiPersonTracker
    from src.ui.visualizer import Visualizer
    from src.audio.alert_coordinator import AlertCoordinator
    from src.audio.alert_engine import AlertEngine

    engine_type = config.get("detection", {}).get("engine", "mobilenet")
    if engine_type == "mobilenet":
        detector = MobileNetPersonDetector(config)
    else:
        detector = PersonDetector(config)
    estimator = DistanceEstimator(config)
    tracker   = MultiPersonTracker(config)
    viz       = Visualizer(config)
    engine    = AlertEngine(config)
    coord     = AlertCoordinator(config, engine)
    engine.start()

    scene   = SyntheticSceneGenerator()
    writer  = None

    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, args.fps, (1280, 720))

    frame_interval = 1.0 / args.fps
    start = time.perf_counter()

    for i in range(args.frames):
        frame = scene.generate_frame(i)
        detections = detector.detect(frame)
        estimates  = [estimator.estimate(d, frame.shape[0]) for d in detections]
        tracks     = tracker.update(detections, estimates, frame.shape[:2])
        coord.evaluate(tracks, frame.shape[1])
        summary    = coord.get_active_alert_summary(tracks)
        elapsed    = time.perf_counter() - start
        fps        = (i + 1) / max(elapsed, 0.001)
        annotated  = viz.draw(frame, tracks, fps, summary, detector.current_fps)

        cv2.imshow("CrowdAware AI — Demo", annotated)
        if writer:
            writer.write(annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

        # Throttle to target FPS
        sleep = frame_interval - (time.perf_counter() - (start + i * frame_interval))
        if sleep > 0:
            time.sleep(sleep)

    engine.stop()
    cv2.destroyAllWindows()
    if writer:
        writer.release()
        print(f"[Demo] Saved output to: {args.save}")

    print(f"\n[Demo] Completed {i+1} frames.")
    print(f"[Demo] Detector avg: {detector.avg_inference_ms:.1f} ms / {detector.current_fps:.1f} FPS")


def run_video_demo(args):
    """Run on a video file."""
    print(f"\n[Demo] Video mode: {args.video}")
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)
    config.setdefault("audio", {})["enabled"] = False
    config["camera"]["device_id"] = args.video

    from src.pipeline import CrowdAwarePipeline
    pipeline = CrowdAwarePipeline(config)
    pipeline.run()


def main():
    args = parse_args()
    try:
        if args.synthetic:
            run_synthetic_demo(args)
        elif args.video:
            run_video_demo(args)
        else:
            print("Please use --synthetic or --video <path>")
    except KeyboardInterrupt:
        print("\n[Demo] Interrupted.")
    except Exception as e:
        print(f"[Demo] Error: {e}")
        raise


if __name__ == "__main__":
    main()
