#!/usr/bin/env python3
"""
calibrate.py
------------
Camera focal length calibration tool for distance estimation.

Usage:
    # Interactive calibration (live camera)
    python scripts/calibrate.py --interactive

    # Manual calibration from known values
    python scripts/calibrate.py --distance 2.0 --height 320

    # Calibrate and auto-update config
    python scripts/calibrate.py --distance 2.0 --height 320 --update-config

How to calibrate:
    1. Stand a person of known height at a measured distance from the camera
    2. Run the system briefly to see their bounding box height in pixels
    3. Run this script with those values
    4. Update focal_length_px in config/settings.yaml

The formula is:
    focal_length_px = (distance_m × bbox_height_px) / person_real_height_m

Author: CrowdAware AI Team
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate the camera focal length for accurate distance estimation."
    )
    parser.add_argument("--distance",  "-d", type=float, help="Known distance to person (metres)")
    parser.add_argument("--height",    "-p", type=int,   help="Measured bounding box height (pixels)")
    parser.add_argument("--person-height", type=float, default=1.70,
                        help="Real height of calibration person in metres (default: 1.70)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run live interactive calibration with camera")
    parser.add_argument("--update-config", action="store_true",
                        help="Automatically write computed focal_length to config/settings.yaml")
    parser.add_argument("--config", default="config/settings.yaml")
    return parser.parse_args()


def compute_focal_length(distance_m: float, bbox_height_px: int, person_height_m: float = 1.70) -> float:
    """
    Compute focal length from a reference measurement.

    Args:
        distance_m       : True distance of the person in metres
        bbox_height_px   : Measured bounding box height in pixels
        person_height_m  : Real height of the person in metres

    Returns:
        focal_length_px
    """
    if bbox_height_px <= 0:
        raise ValueError("bbox_height_px must be > 0")
    if distance_m <= 0:
        raise ValueError("distance_m must be > 0")
    return (distance_m * bbox_height_px) / person_height_m


def update_config(config_path: str, focal_length: float):
    """Write the computed focal_length_px into the YAML config."""
    try:
        import yaml
        path = Path(config_path)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("distance", {})["focal_length_px"] = round(focal_length, 2)
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Updated focal_length_px = {focal_length:.2f} in {config_path}")
    except Exception as e:
        print(f"✗ Failed to update config: {e}")


def interactive_calibration():
    """
    Live camera calibration:
    Display live feed, let user position a known person, read bbox height.
    """
    try:
        import cv2
        import yaml
        import sys
        sys.path.insert(0, ".")
        with open("config/settings.yaml") as f:
            config = yaml.safe_load(f)

        from src.detection.detector import PersonDetector
        detector = PersonDetector(config)

        cap = cv2.VideoCapture(config.get("camera", {}).get("device_id", 0))
        if not cap.isOpened():
            print("Cannot open camera.")
            return

        print("\n=== Interactive Calibration Mode ===")
        print("Stand a person at a KNOWN measured distance from the camera.")
        print("Press SPACE to capture the measurement.")
        print("Press Q to quit without saving.\n")

        known_distance = float(input("Enter the known distance in metres (e.g. 2.0): "))
        person_height  = float(input("Enter the person's real height in metres (default 1.70): ") or "1.70")

        measurements = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"H={det.height}px", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            cv2.putText(frame, "SPACE=Capture  Q=Quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.imshow("Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                if detections:
                    h_px = detections[0].height
                    fl = compute_focal_length(known_distance, h_px, person_height)
                    measurements.append(fl)
                    print(f"  Captured: bbox_height={h_px}px → focal_length={fl:.1f}px")
                else:
                    print("  No person detected — try again.")

        cap.release()
        cv2.destroyAllWindows()

        if measurements:
            avg_fl = sum(measurements) / len(measurements)
            print(f"\n=== Calibration Results ===")
            print(f"  Measurements : {len(measurements)}")
            print(f"  Average focal length: {avg_fl:.2f} px")
            print(f"\n  → Set focal_length_px: {avg_fl:.2f} in config/settings.yaml")
            update = input("\nUpdate config automatically? [y/N]: ").strip().lower()
            if update == "y":
                update_config("config/settings.yaml", avg_fl)

    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")


def main():
    args = parse_args()

    if args.interactive:
        interactive_calibration()
        return

    if args.distance is None or args.height is None:
        print("Error: Provide --distance and --height for manual calibration, "
              "or use --interactive for live calibration.")
        print("\nExample:")
        print("  python scripts/calibrate.py --distance 2.0 --height 320")
        sys.exit(1)

    try:
        focal = compute_focal_length(args.distance, args.height, args.person_height)
    except ValueError as e:
        print(f"Calibration error: {e}")
        sys.exit(1)

    print("\n=== Calibration Result ===")
    print(f"  Known distance    : {args.distance} m")
    print(f"  BBox height       : {args.height} px")
    print(f"  Person height     : {args.person_height} m")
    print(f"  ─────────────────────────────")
    print(f"  focal_length_px   : {focal:.2f}")
    print()
    print("  Update config/settings.yaml:")
    print(f"    distance:")
    print(f"      focal_length_px: {focal:.2f}")

    # Verify estimate at 1m, 2m, 3m, 5m
    ph = args.person_height
    print("\n  Distance verification table:")
    print(f"  {'Distance (m)':>14} | {'BBox Height (px)':>17}")
    print(f"  {'-'*14}-+-{'-'*17}")
    for d in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0]:
        expected_h = (focal * ph) / d
        print(f"  {d:>14.1f} | {expected_h:>17.0f}")

    if args.update_config:
        update_config(args.config, focal)


if __name__ == "__main__":
    main()
