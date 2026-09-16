"""
pipeline.py
-----------
Main application pipeline — orchestrates the full processing loop:

    Camera → Frame → Detect → Estimate Distance → Track → Alert → Visualize

Runs the camera capture, detection, tracking, and alert systems
in a coordinated single-thread loop with optional background
detection threading for higher camera FPS.

Author: CrowdAware AI Team
"""

import logging
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detection.detector import PersonDetector, Detection
from .detection.distance import DistanceEstimator, DistanceEstimate
from .detection.tracker import MultiPersonTracker, Track
from .audio.alert_engine import AlertEngine
from .audio.alert_coordinator import AlertCoordinator
from .ui.visualizer import Visualizer

logger = logging.getLogger(__name__)


class CrowdAwarePipeline:
    """
    Top-level orchestrator for the CrowdAware AI system.

    Initializes and connects all subsystems, then runs the
    main event loop until the user quits.

    Usage
    -----
        pipeline = CrowdAwarePipeline(config)
        pipeline.run()
    """

    def __init__(self, config: dict):
        self.config = config
        cam_cfg  = config.get("camera", {})
        disp_cfg = config.get("display", {})
        perf_cfg = config.get("performance", {})

        # Camera settings
        self.cam_id     = cam_cfg.get("device_id",   0)
        self.cam_width  = cam_cfg.get("width",        1280)
        self.cam_height = cam_cfg.get("height",       720)
        self.cam_fps    = cam_cfg.get("fps",          30)
        self.cam_flip   = cam_cfg.get("flip_horizontal", False)

        # Performance
        self.skip_frames    = perf_cfg.get("skip_frames", 0)
        self.max_det_fps    = perf_cfg.get("max_detection_fps", 15)

        # Display
        self.window_name = disp_cfg.get("window_name", "CrowdAware AI")
        self.show_window = disp_cfg.get("enabled", True)

        # Subsystems
        logger.info("[Pipeline] Initializing subsystems…")
        self.detector    = PersonDetector(config)
        self.estimator   = DistanceEstimator(config)
        self.tracker     = MultiPersonTracker(config)
        self.alert_engine = AlertEngine(config)
        self.coordinator = AlertCoordinator(config, self.alert_engine)
        self.visualizer  = Visualizer(config)

        # State
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._frame_count = 0
        self._det_frame_count = 0

        # FPS smoothing
        self._cam_fps_history = []
        self._det_fps_history = []
        self._last_frame_time = 0.0
        self._last_det_time   = 0.0
        self._det_min_interval = 1.0 / max(self.max_det_fps, 1)

        logger.info("[Pipeline] All subsystems initialized.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self):
        """
        Open the camera and run the main processing loop.
        Press Q or ESC to quit.
        """
        self._cap = self._open_camera()
        if self._cap is None:
            logger.error("[Pipeline] Cannot open camera — aborting.")
            return

        self.alert_engine.start()
        self.alert_engine.force_alert("CrowdAware system started. Monitoring for nearby people.")

        self._running = True
        self._last_frame_time = time.perf_counter()
        logger.info("[Pipeline] ▶  Main loop started. Press Q or ESC to quit.")

        try:
            while self._running:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    logger.warning("[Pipeline] Failed to read frame — retrying…")
                    time.sleep(0.05)
                    continue

                if self.cam_flip:
                    frame = cv2.flip(frame, 1)

                self._frame_count += 1
                cam_fps = self._update_cam_fps()

                # Throttle detection (don't run inference on every frame)
                now = time.perf_counter()
                run_detection = (now - self._last_det_time) >= self._det_min_interval
                if self.skip_frames > 0:
                    run_detection = run_detection and (self._frame_count % (self.skip_frames + 1) == 0)

                tracks = self._last_tracks if hasattr(self, "_last_tracks") else []

                if run_detection:
                    tracks = self._process_frame(frame)
                    self._last_tracks = tracks
                    self._last_det_time = now
                    self._det_frame_count += 1

                det_fps = self._update_det_fps()

                # --- Alert evaluation ---
                h, w = frame.shape[:2]
                self.coordinator.evaluate(tracks, w)
                alert_summary = self.coordinator.get_active_alert_summary(tracks)

                # --- Visualization ---
                if self.show_window:
                    annotated = self.visualizer.draw(
                        frame, tracks, cam_fps, alert_summary, det_fps
                    )
                    cv2.imshow(self.window_name, annotated)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):   # Q or ESC
                        logger.info("[Pipeline] Quit requested by user.")
                        break
                    elif key == ord("c"):
                        self._run_calibration_hint(frame)
                    elif key == ord("s"):
                        self._save_screenshot(annotated)
                    elif key == ord("a"):
                        self.alert_engine.force_alert("Manual test alert.")

        except KeyboardInterrupt:
            logger.info("[Pipeline] KeyboardInterrupt — shutting down.")
        finally:
            self.stop()

    def stop(self):
        """Release all resources."""
        self._running = False
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()
        self.alert_engine.stop()

        # Print final stats
        self._print_stats()
        logger.info("[Pipeline] ✓ Shutdown complete.")

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> List[Track]:
        """
        Run detection → distance estimation → tracking on one frame.
        Returns the updated list of active Track objects.
        """
        h, w = frame.shape[:2]

        # 1. Detect persons
        detections: List[Detection] = self.detector.detect(frame)

        # 2. Estimate distance for each detection
        estimates: List[Optional[DistanceEstimate]] = [
            self.estimator.estimate(det, h) for det in detections
        ]

        # 3. Update tracker
        tracks: List[Track] = self.tracker.update(detections, estimates, (h, w))

        logger.debug(
            f"[Pipeline] Frame {self._frame_count}: "
            f"{len(detections)} detections, {len(tracks)} tracks"
        )
        return tracks

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        """Open the camera and configure resolution / FPS."""
        logger.info(f"[Pipeline] Opening camera device {self.cam_id}…")
        cap = cv2.VideoCapture(self.cam_id)
        if not cap.isOpened():
            logger.error(f"[Pipeline] Cannot open camera {self.cam_id}")
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        cap.set(cv2.CAP_PROP_FPS,          self.cam_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # Minimize latency

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"[Pipeline] Camera opened: {actual_w}×{actual_h} @ {actual_fps:.0f} fps")
        return cap

    # ------------------------------------------------------------------
    # FPS tracking
    # ------------------------------------------------------------------

    def _update_cam_fps(self) -> float:
        now = time.perf_counter()
        dt  = now - self._last_frame_time
        self._last_frame_time = now
        fps = 1.0 / dt if dt > 0 else 0.0
        self._cam_fps_history.append(fps)
        if len(self._cam_fps_history) > 30:
            self._cam_fps_history.pop(0)
        return sum(self._cam_fps_history) / len(self._cam_fps_history)

    def _update_det_fps(self) -> float:
        fps = self.detector.current_fps
        self._det_fps_history.append(fps)
        if len(self._det_fps_history) > 30:
            self._det_fps_history.pop(0)
        return sum(self._det_fps_history) / len(self._det_fps_history)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _run_calibration_hint(self, frame: np.ndarray):
        """Print calibration instructions when user presses C."""
        h = frame.shape[0]
        logger.info(
            "[Pipeline] Calibration mode hint:\n"
            "  Stand a known person at a measured distance (e.g. 2.0 m).\n"
            "  Note the bounding box height in pixels from the console.\n"
            "  Run: python scripts/calibrate.py --distance 2.0 --height <px>\n"
            "  Update focal_length_px in config/settings.yaml."
        )
        print("\n[C] Calibration: stand someone at a known distance and run:")
        print("    python scripts/calibrate.py --distance <metres> --height <bbox_px>\n")

    def _save_screenshot(self, frame: np.ndarray):
        """Save current frame as PNG."""
        import os
        os.makedirs("screenshots", exist_ok=True)
        filename = f"screenshots/frame_{self._frame_count:06d}.png"
        cv2.imwrite(filename, frame)
        logger.info(f"[Pipeline] Screenshot saved → {filename}")
        print(f"[S] Screenshot saved: {filename}")

    def _print_stats(self):
        """Print end-of-session statistics."""
        det_stats   = self.detector.get_stats()
        track_stats = self.tracker.get_stats()
        audio_stats = self.alert_engine.get_stats()
        print("\n" + "="*50)
        print("  CrowdAware AI — Session Statistics")
        print("="*50)
        print(f"  Frames captured    : {self._frame_count}")
        print(f"  Frames processed   : {det_stats['frames_processed']}")
        print(f"  Total detections   : {det_stats['total_detections']}")
        print(f"  Avg inference time : {det_stats['avg_inference_ms']:.1f} ms")
        print(f"  Model used         : {det_stats['model']} ({det_stats['device']})")
        print(f"  Alerts issued      : {audio_stats['alerts_issued']}")
        print(f"  Alerts suppressed  : {audio_stats['alerts_suppressed']}")
        print("="*50 + "\n")
