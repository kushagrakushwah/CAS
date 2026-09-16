"""
visualizer.py
-------------
Real-time visualization overlay for the CrowdAware system.

Draws on each video frame:
- Color-coded bounding boxes by zone (Critical=Red, Close=Orange, ...)
- Track IDs and distance labels
- Trajectory trails (position history)
- Direction arrows (movement vectors)
- Central path corridor overlay
- Dashboard panel (stats, alerts, crowd count)
- FPS counter

All drawing is done with OpenCV (cv2) for maximum performance.

Author: CrowdAware AI Team
"""

import logging
import math
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..detection.tracker import Track
from ..detection.distance import Zone, DistanceEstimator

logger = logging.getLogger(__name__)

# BGR color palette
ZONE_COLORS = {
    Zone.CRITICAL : (0,   0,   255),
    Zone.CLOSE    : (0,   100, 255),
    Zone.NEAR     : (0,   220, 220),
    Zone.MEDIUM   : (0,   200, 50),
    Zone.FAR      : (200, 180, 0),
}

DARK_BG     = (18, 18, 18)
WHITE       = (255, 255, 255)
LIGHT_GREY  = (180, 180, 180)
MID_GREY    = (100, 100, 100)
ACCENT_CYAN = (255, 220, 50)
ARROW_COLOR = (50,  220, 255)
TRAJ_COLOR  = (200, 120, 50)


class Visualizer:
    """
    Renders all visual overlays onto each frame.

    Design
    ------
    - Dark, high-contrast UI suitable for outdoor use
    - Minimal text on the camera feed; rich info in side dashboard
    - Trajectory trails fade toward older positions
    - Direction arrows scale with speed
    - Semi-transparent path corridor overlay

    Usage
    -----
        viz = Visualizer(config)
        annotated_frame = viz.draw(frame, tracks, fps, alert_summary)
    """

    def __init__(self, config: dict):
        disp_cfg  = config.get("display", {})
        zone_cfg  = config.get("zones", {})
        track_cfg = config.get("tracking", {})

        self.show_fps        = disp_cfg.get("show_fps",           True)
        self.show_distance   = disp_cfg.get("show_distance",      True)
        self.show_direction  = disp_cfg.get("show_direction",     True)
        self.show_track_id   = disp_cfg.get("show_track_id",      True)
        self.show_trajectory = disp_cfg.get("show_trajectory",    True)
        self.traj_len        = disp_cfg.get("trajectory_length",  20)
        self.show_corridor   = disp_cfg.get("show_path_corridor", True)
        self.show_dashboard  = disp_cfg.get("show_dashboard",     True)
        self.dash_width      = disp_cfg.get("dashboard_width",    300)

        self.path_frac = float(zone_cfg.get("path_center_fraction", 0.40))

        # Font settings
        self.font       = cv2.FONT_HERSHEY_SIMPLEX
        self.font_small = 0.45
        self.font_med   = 0.60
        self.font_large = 0.85
        self.thick_thin = 1
        self.thick_med  = 2

        # FPS smoothing
        self._fps_history = []
        self._last_time   = time.perf_counter()

        logger.info(f"[Visualizer] Initialized. Dashboard={self.show_dashboard}, "
                    f"Trajectory={self.show_trajectory}")

    # ------------------------------------------------------------------
    # Main draw method
    # ------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        fps: float = 0.0,
        alert_summary: str = "",
        detector_fps: float = 0.0,
    ) -> np.ndarray:
        """
        Annotate a video frame with all overlays.

        Args:
            frame         : BGR camera frame
            tracks        : List of active Track objects
            fps           : Current capture FPS
            alert_summary : One-line alert status string for dashboard
            detector_fps  : Inference FPS from detector

        Returns:
            Annotated BGR frame (same dimensions as input)
        """
        if frame is None:
            return frame

        h, w = frame.shape[:2]
        out = frame.copy()

        # 1. Semi-transparent path corridor
        if self.show_corridor:
            out = self._draw_corridor(out, w, h)

        # 2. Per-track overlays
        for track in tracks:
            out = self._draw_track(out, track, w, h)

        # 3. FPS counter
        if self.show_fps:
            self._draw_fps(out, fps, detector_fps)

        # 4. Crowd count badge
        self._draw_crowd_badge(out, len(tracks))

        # 5. Dashboard side panel
        if self.show_dashboard:
            out = self._draw_dashboard(out, tracks, alert_summary, fps, detector_fps)

        return out

    # ------------------------------------------------------------------
    # Path corridor overlay
    # ------------------------------------------------------------------

    def _draw_corridor(self, frame: np.ndarray, w: int, h: int) -> np.ndarray:
        """Draw a semi-transparent yellow corridor in the center of frame."""
        overlay = frame.copy()
        pad_x = int(w * (0.5 - self.path_frac / 2))
        cv2.rectangle(overlay, (pad_x, 0), (w - pad_x, h), (0, 220, 255), -1)
        return cv2.addWeighted(overlay, 0.07, frame, 0.93, 0)

    # ------------------------------------------------------------------
    # Per-track drawing
    # ------------------------------------------------------------------

    def _draw_track(self, frame: np.ndarray, track: Track, fw: int, fh: int) -> np.ndarray:
        """Draw all elements for one tracked person."""
        x1, y1, x2, y2 = track.bbox
        zone  = track.zone or Zone.FAR
        color = ZONE_COLORS.get(zone, (200, 200, 200))

        # Clamp to frame
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(fw, x2); y2 = min(fh, y2)

        # --- Bounding box (thickness increases for closer zones) ---
        thickness = {
            Zone.CRITICAL: 3,
            Zone.CLOSE:    2,
            Zone.NEAR:     2,
            Zone.MEDIUM:   1,
            Zone.FAR:      1,
        }.get(zone, 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # --- Corner accents (bracket style) ---
        self._draw_corner_brackets(frame, x1, y1, x2, y2, color)

        # --- Top label bar ---
        label_parts = []
        if self.show_track_id:
            label_parts.append(f"#{track.track_id}")
        if self.show_distance and track.distance_estimate:
            label_parts.append(f"{track.distance_m:.1f}m")
        if label_parts:
            label = "  ".join(label_parts)
            self._put_label(frame, label, x1, y1 - 4, color)

        # --- Zone badge ---
        zone_text = zone.value.upper()
        self._put_small_badge(frame, zone_text, x2 - 5, y1 + 5, color, anchor="tr")

        # --- Approaching / path blocker indicators ---
        if track.is_approaching:
            self._draw_approach_indicator(frame, x1, y1, x2, y2, color)
        if track.is_path_blocker:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # --- Direction arrow ---
        if self.show_direction and not track.is_stationary:
            self._draw_direction_arrow(frame, track)

        # --- Trajectory trail ---
        if self.show_trajectory and len(track.center_history) > 2:
            self._draw_trajectory(frame, track)

        return frame

    def _draw_corner_brackets(
        self, frame, x1, y1, x2, y2, color, length: int = 14
    ):
        """Draw L-shaped corner brackets inside bbox corners."""
        lw = 2
        # TL
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, lw)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, lw)
        # TR
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, lw)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, lw)
        # BL
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, lw)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, lw)
        # BR
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, lw)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, lw)

    def _draw_approach_indicator(self, frame, x1, y1, x2, y2, color):
        """Pulsing triangle indicator at bottom of bbox when approaching."""
        cx = (x1 + x2) // 2
        pts = np.array([[cx, y2 + 18], [cx - 8, y2 + 4], [cx + 8, y2 + 4]], np.int32)
        cv2.fillPoly(frame, [pts], color)
        cv2.putText(frame, "APPR", (cx - 16, y2 + 30),
                    self.font, 0.35, color, 1, cv2.LINE_AA)

    def _draw_direction_arrow(self, frame: np.ndarray, track: Track):
        """Draw a motion arrow at the person's center."""
        if len(track.center_history) < 3:
            return
        hist = list(track.center_history)
        old  = hist[max(0, len(hist) - 8)]
        new  = hist[-1]
        dx   = new[0] - old[0]
        dy   = new[1] - old[1]
        mag  = math.sqrt(dx**2 + dy**2)
        if mag < 3:
            return
        # Scale arrow length
        scale  = min(40, mag * 2.5)
        end_x  = int(new[0] + (dx / mag) * scale)
        end_y  = int(new[1] + (dy / mag) * scale)
        cv2.arrowedLine(frame, new, (end_x, end_y), ARROW_COLOR, 2,
                        tipLength=0.35, line_type=cv2.LINE_AA)

    def _draw_trajectory(self, frame: np.ndarray, track: Track):
        """Draw fading trajectory trail behind person."""
        hist = list(track.center_history)[-self.traj_len:]
        for i in range(1, len(hist)):
            alpha = i / len(hist)   # newer = brighter
            r = int(TRAJ_COLOR[0] * alpha)
            g = int(TRAJ_COLOR[1] * alpha)
            b = int(TRAJ_COLOR[2] * alpha)
            thickness = max(1, int(alpha * 3))
            cv2.line(frame, hist[i - 1], hist[i], (b, g, r), thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # HUD elements
    # ------------------------------------------------------------------

    def _draw_fps(self, frame: np.ndarray, cam_fps: float, det_fps: float):
        """Draw FPS counter in top-left corner."""
        txt = f"CAM {cam_fps:.0f} fps  |  DET {det_fps:.0f} fps"
        self._put_label_plain(frame, txt, 10, 22, color=ACCENT_CYAN)

    def _draw_crowd_badge(self, frame: np.ndarray, count: int):
        """Draw person count badge top-right."""
        h, w = frame.shape[:2]
        txt = f"PEOPLE: {count}"
        color = (0, 200, 0) if count < 5 else (0, 100, 255) if count < 10 else (0, 0, 255)
        tw, th = cv2.getTextSize(txt, self.font, self.font_med, 2)[0]
        x = w - tw - 15
        cv2.rectangle(frame, (x - 6, 8), (w - 8, 8 + th + 10), DARK_BG, -1)
        cv2.putText(frame, txt, (x, 8 + th + 2), self.font, self.font_med, color, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Dashboard panel
    # ------------------------------------------------------------------

    def _draw_dashboard(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        alert_summary: str,
        fps: float,
        det_fps: float,
    ) -> np.ndarray:
        """Draw a semi-opaque side panel with track list and stats."""
        h, w = frame.shape[:2]
        dash_x = w - self.dash_width

        # Dark overlay panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (dash_x, 0), (w, h), (10, 10, 10), -1)
        frame = cv2.addWeighted(overlay, 0.70, frame, 0.30, 0)

        y = 20
        self._put_label_plain(frame, "CROWDAWARE AI", dash_x + 10, y, WHITE, scale=0.65, thick=2)
        y += 5
        cv2.line(frame, (dash_x + 8, y), (w - 8, y), MID_GREY, 1)
        y += 18

        # Alert summary
        alert_col = (0, 0, 255) if "CRITICAL" in alert_summary.upper() else \
                    (0, 150, 255) if "CLOSE" in alert_summary.upper() else \
                    (0, 200, 200) if "BLOCKED" in alert_summary.upper() else \
                    (0, 200, 80)
        for line in self._wrap_text(alert_summary, 28):
            self._put_label_plain(frame, line, dash_x + 10, y, alert_col, scale=0.45)
            y += 16
        y += 6

        cv2.line(frame, (dash_x + 8, y), (w - 8, y), MID_GREY, 1)
        y += 14

        # Track table
        self._put_label_plain(frame, "TRACKED PERSONS", dash_x + 10, y, LIGHT_GREY, scale=0.42, thick=1)
        y += 14
        header = f"{'ID':>4}  {'Dist':>6}  {'Zone':>8}  {'State'}"
        self._put_label_plain(frame, header, dash_x + 8, y, MID_GREY, scale=0.36)
        y += 13
        cv2.line(frame, (dash_x + 8, y), (w - 8, y), (40, 40, 40), 1)
        y += 10

        sorted_tracks = sorted(tracks, key=lambda t: t.distance_m)
        for track in sorted_tracks[:10]:   # max 10 rows
            tid   = f"#{track.track_id}"
            dist  = f"{track.distance_m:.1f}m"
            zone  = track.zone.value[:6] if track.zone else "far"
            state = "APPR" if track.is_approaching else \
                    "BLCK" if track.is_path_blocker else \
                    "STAT" if track.is_stationary else "MOVE"
            row = f"{tid:>4}  {dist:>6}  {zone:>8}  {state}"
            color = ZONE_COLORS.get(track.zone, LIGHT_GREY)
            self._put_label_plain(frame, row, dash_x + 8, y, color, scale=0.38)
            y += 14

        # Stats footer
        y = h - 70
        cv2.line(frame, (dash_x + 8, y), (w - 8, y), MID_GREY, 1)
        y += 14
        self._put_label_plain(frame, f"Cam FPS : {fps:.0f}", dash_x + 10, y, MID_GREY, scale=0.40)
        y += 14
        self._put_label_plain(frame, f"Det FPS : {det_fps:.0f}", dash_x + 10, y, MID_GREY, scale=0.40)
        y += 14
        count = len(tracks)
        density = "LOW" if count < 5 else "MED" if count < 10 else "HIGH"
        dcol = (0, 200, 0) if density == "LOW" else (0, 160, 255) if density == "MED" else (0, 0, 255)
        self._put_label_plain(frame, f"Density : {density} ({count})", dash_x + 10, y, dcol, scale=0.40)

        return frame

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    def _put_label(
        self, frame: np.ndarray, text: str, x: int, y: int,
        color: Tuple[int,int,int], scale: float = 0.50
    ):
        """Draw text with dark background pill."""
        (tw, th), _ = cv2.getTextSize(text, self.font, scale, 1)
        cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 4, y + 2), DARK_BG, -1)
        cv2.putText(frame, text, (x, y), self.font, scale, color, 1, cv2.LINE_AA)

    def _put_label_plain(
        self, frame: np.ndarray, text: str, x: int, y: int,
        color: Tuple[int,int,int] = WHITE, scale: float = 0.50, thick: int = 1
    ):
        cv2.putText(frame, text, (x, y), self.font, scale, color, thick, cv2.LINE_AA)

    def _put_small_badge(
        self, frame: np.ndarray, text: str, x: int, y: int,
        color: Tuple[int,int,int], anchor: str = "tl"
    ):
        """Draw a small filled badge."""
        (tw, th), _ = cv2.getTextSize(text, self.font, 0.35, 1)
        if anchor == "tr":
            x -= (tw + 6)
        cv2.rectangle(frame, (x, y), (x + tw + 6, y + th + 6), color, -1)
        cv2.putText(frame, text, (x + 3, y + th + 2), self.font, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    @staticmethod
    def _wrap_text(text: str, width: int) -> List[str]:
        """Wrap text to `width` characters per line."""
        words = text.split()
        lines, current = [], ""
        for word in words:
            if len(current) + len(word) + 1 <= width:
                current = (current + " " + word).strip()
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [""]
