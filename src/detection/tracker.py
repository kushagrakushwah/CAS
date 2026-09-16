"""
tracker.py
----------
Multi-person tracker using Kalman filtering and Hungarian-algorithm assignment.

Each track maintains:
- Unique ID
- Kalman-filtered position/velocity state
- Position history for movement analysis
- Distance and zone history

This is a SORT-inspired tracker (Simple Online and Realtime Tracking),
simplified and adapted for this use case.

Author: CrowdAware AI Team
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detector import Detection
from .distance import DistanceEstimate, Zone

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Kalman Filter for 2D bounding box tracking
# ------------------------------------------------------------------

class KalmanBoxTracker:
    """
    Kalman filter for tracking bounding boxes.

    State vector: [x, y, s, r, dx, dy, ds]
        x, y  = center position
        s     = scale (area)
        r     = aspect ratio (width/height, constant)
        dx,dy = velocity in x,y
        ds    = rate of change of scale

    Observation: [x, y, s, r]
    """

    _id_counter = 0

    @classmethod
    def _next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    def __init__(self, bbox: List[int]):
        """
        Initialize tracker with an initial bounding box [x1, y1, x2, y2].
        """
        self.id = self._next_id()
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0

        # Build Kalman filter matrices
        dt = 1.0   # time step (frames)

        # State transition (constant velocity model)
        self.F = np.array([
            [1, 0, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0, 0,  dt, 0 ],
            [0, 0, 1, 0, 0,  0,  dt],
            [0, 0, 0, 1, 0,  0,  0 ],
            [0, 0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0, 0,  0,  1 ],
        ], dtype=float)

        # Measurement matrix (observe x,y,s,r)
        self.H = np.zeros((4, 7), dtype=float)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        # Measurement noise
        self.R = np.eye(4, dtype=float) * 1.0
        self.R[2, 2] *= 10
        self.R[3, 3] *= 10

        # Process noise
        self.Q = np.eye(7, dtype=float) * 0.01
        self.Q[4:, 4:] *= 0.01

        # State covariance
        self.P = np.eye(7, dtype=float) * 10.0
        self.P[4:, 4:] *= 1000.0

        # Initial state
        self.x = self._bbox_to_z(bbox)
        # Extend to 7-dim state (velocity = 0)
        state = np.zeros((7, 1))
        state[:4] = self.x
        self.x = state

    # ---------------------------------------------------------------
    # Kalman steps
    # ---------------------------------------------------------------

    def predict(self) -> np.ndarray:
        """Advance state estimate one time step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self._x_to_bbox()

    def update(self, bbox: List[int]):
        """Incorporate a new observation."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        z = self._bbox_to_z(bbox)
        y = z - self.H @ self.x             # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _bbox_to_z(bbox: List[int]) -> np.ndarray:
        """Convert [x1,y1,x2,y2] to measurement vector [cx,cy,s,r]."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        s  = float((x2 - x1) * (y2 - y1))   # area
        r  = float((x2 - x1) / max(y2 - y1, 1))  # aspect ratio
        return np.array([[cx], [cy], [s], [r]], dtype=float)

    def _x_to_bbox(self) -> List[int]:
        """Convert state vector back to [x1,y1,x2,y2]."""
        cx, cy, s, r = float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0])
        s = max(s, 1.0)
        w = np.sqrt(s * r)
        h = s / max(w, 1.0)
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        return [x1, y1, x2, y2]

    def get_state(self) -> List[int]:
        return self._x_to_bbox()

    @property
    def center(self) -> Tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.x[4, 0]), float(self.x[5, 0])


# ------------------------------------------------------------------
# Track — combines Kalman tracker with rich metadata
# ------------------------------------------------------------------

@dataclass
class Track:
    """
    A confirmed or tentative track for a single person.

    Stores filtered position, velocity, distance history,
    and movement analysis results.
    """
    track_id: int
    kalman: KalmanBoxTracker

    # Latest observations
    bbox: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    confidence: float = 0.0
    distance_estimate: Optional[DistanceEstimate] = None

    # History buffers
    center_history: deque = field(default_factory=lambda: deque(maxlen=30))
    distance_history: deque = field(default_factory=lambda: deque(maxlen=30))

    # Movement analysis results (updated each frame)
    speed_mps: float = 0.0             # Speed in metres per second
    approach_rate: float = 0.0         # Positive = retreating, Negative = approaching
    movement_direction: str = "unknown"
    is_approaching: bool = False
    is_retreating: bool = False
    is_stationary: bool = True
    is_path_blocker: bool = False

    # Track lifecycle
    age: int = 0
    consecutive_misses: int = 0

    def update_history(self):
        """Append current center and distance to history deques."""
        cx = (self.bbox[0] + self.bbox[2]) // 2
        cy = (self.bbox[1] + self.bbox[3]) // 2
        self.center_history.append((cx, cy))
        if self.distance_estimate:
            self.distance_history.append(self.distance_estimate.distance_m)

    @property
    def zone(self) -> Optional[Zone]:
        return self.distance_estimate.zone if self.distance_estimate else None

    @property
    def distance_m(self) -> float:
        return self.distance_estimate.distance_m if self.distance_estimate else 999.0

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.bbox[0] + self.bbox[2]) // 2,
                (self.bbox[1] + self.bbox[3]) // 2)

    def __repr__(self) -> str:
        return (f"Track(id={self.track_id}, dist={self.distance_m:.1f}m, "
                f"zone={self.zone}, approach={self.is_approaching})")


# ------------------------------------------------------------------
# Multi-Person Tracker
# ------------------------------------------------------------------

class MultiPersonTracker:
    """
    Tracks multiple people across video frames.

    Algorithm:
    1. Predict next position of all existing tracks (Kalman predict step)
    2. Compute IoU matrix between predicted tracks and new detections
    3. Run Hungarian algorithm to find optimal assignments
    4. Update matched tracks, create new tracks for unmatched detections
    5. Increment miss count for unmatched tracks and delete if too old

    Usage:
        tracker = MultiPersonTracker(config)
        tracks  = tracker.update(detections, distance_estimates, frame_shape)
    """

    def __init__(self, config: dict):
        track_cfg  = config.get("tracking", {})
        move_cfg   = config.get("movement", {})
        dist_cfg   = config.get("distance", {})

        self.max_age       = track_cfg.get("max_age", 30)
        self.min_hits      = track_cfg.get("min_hits", 3)
        self.iou_threshold = track_cfg.get("iou_threshold", 0.3)

        self.min_speed    = move_cfg.get("min_speed_threshold", 0.05)
        self.approach_thr = move_cfg.get("approach_threshold", -0.15)
        self.retreat_thr  = move_cfg.get("retreat_threshold", 0.15)
        self.hist_frames  = move_cfg.get("history_frames", 15)

        self.path_center_frac  = config.get("zones", {}).get("path_center_fraction", 0.40)
        self.path_block_thr    = config.get("zones", {}).get("path_block_threshold", 0.35)

        self._tracks: Dict[int, Track] = {}
        self._frame_num = 0

        logger.info(
            f"[Tracker] max_age={self.max_age}, min_hits={self.min_hits}, "
            f"iou_thr={self.iou_threshold}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Detection],
        distance_estimates: List[Optional[DistanceEstimate]],
        frame_shape: Tuple[int, int],
    ) -> List[Track]:
        """
        Update all tracks with new detections.

        Args:
            detections        : List of Detection objects from detector
            distance_estimates: Parallel list of DistanceEstimate (or None)
            frame_shape       : (height, width) of current frame

        Returns:
            List of active, confirmed Track objects.
        """
        self._frame_num += 1
        frame_h, frame_w = frame_shape

        # --- Step 1: Predict ---
        predicted_bboxes = {}
        for tid, track in list(self._tracks.items()):
            predicted_bboxes[tid] = track.kalman.predict()

        # --- Step 2: Compute IoU matrix ---
        track_ids = list(self._tracks.keys())
        matched, unmatched_dets, unmatched_trks = self._associate(
            detections, track_ids, predicted_bboxes
        )

        # --- Step 3: Update matched tracks ---
        for det_idx, trk_id in matched:
            det  = detections[det_idx]
            dist = distance_estimates[det_idx] if det_idx < len(distance_estimates) else None
            track = self._tracks[trk_id]
            track.kalman.update(det.bbox)
            track.bbox = track.kalman.get_state()
            track.confidence = det.confidence
            track.distance_estimate = dist
            track.age = track.kalman.age
            track.consecutive_misses = 0
            track.update_history()
            self._analyze_movement(track, frame_w)

        # --- Step 4: Create new tracks for unmatched detections ---
        for det_idx in unmatched_dets:
            det  = detections[det_idx]
            dist = distance_estimates[det_idx] if det_idx < len(distance_estimates) else None
            kf   = KalmanBoxTracker(det.bbox)
            new_track = Track(
                track_id=kf.id,
                kalman=kf,
                bbox=det.bbox,
                confidence=det.confidence,
                distance_estimate=dist,
            )
            new_track.update_history()
            self._tracks[kf.id] = new_track

        # --- Step 5: Mark unmatched tracks as missed ---
        for trk_id in unmatched_trks:
            self._tracks[trk_id].consecutive_misses += 1
            self._tracks[trk_id].bbox = predicted_bboxes.get(trk_id, self._tracks[trk_id].bbox)

        # --- Step 6: Remove stale tracks ---
        self._tracks = {
            tid: t for tid, t in self._tracks.items()
            if t.consecutive_misses <= self.max_age
        }

        # Return confirmed tracks only
        return [
            t for t in self._tracks.values()
            if t.kalman.hit_streak >= self.min_hits or
               (t.kalman.hits >= 1 and t.kalman.time_since_update == 0)
        ]

    # ------------------------------------------------------------------
    # Association
    # ------------------------------------------------------------------

    def _associate(
        self,
        detections: List[Detection],
        track_ids: List[int],
        predicted: Dict[int, List[int]],
    ) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
        """
        Match detections to tracks using IoU + Hungarian algorithm.
        Returns: (matched pairs, unmatched det indices, unmatched track IDs)
        """
        if not track_ids or not detections:
            return [], list(range(len(detections))), track_ids

        iou_matrix = np.zeros((len(detections), len(track_ids)), dtype=float)
        for d_idx, det in enumerate(detections):
            for t_idx, tid in enumerate(track_ids):
                pred_box = predicted[tid]
                iou_matrix[d_idx, t_idx] = self._iou(det.bbox, pred_box)

        # Hungarian assignment via scipy
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        except ImportError:
            # Greedy fallback
            row_ind, col_ind = self._greedy_assign(iou_matrix)

        matched, unmatched_dets, unmatched_trks = [], [], []

        matched_trk_indices = set()
        matched_det_indices = set()

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched.append((r, track_ids[c]))
                matched_det_indices.add(r)
                matched_trk_indices.add(c)

        unmatched_dets = [i for i in range(len(detections)) if i not in matched_det_indices]
        unmatched_trks = [track_ids[i] for i in range(len(track_ids)) if i not in matched_trk_indices]

        return matched, unmatched_dets, unmatched_trks

    @staticmethod
    def _iou(b1: List[int], b2: List[int]) -> float:
        """Compute Intersection-over-Union of two bboxes [x1,y1,x2,y2]."""
        xi1 = max(b1[0], b2[0])
        yi1 = max(b1[1], b2[1])
        xi2 = min(b1[2], b2[2])
        yi2 = min(b1[3], b2[3])
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = max(0, b1[2]-b1[0]) * max(0, b1[3]-b1[1])
        area2 = max(0, b2[2]-b2[0]) * max(0, b2[3]-b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _greedy_assign(iou_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple greedy assignment (fallback when scipy is unavailable)."""
        rows, cols = [], []
        used_r, used_c = set(), set()
        flat = np.argsort(-iou_matrix.ravel())
        for idx in flat:
            r, c = divmod(int(idx), iou_matrix.shape[1])
            if r not in used_r and c not in used_c:
                rows.append(r)
                cols.append(c)
                used_r.add(r)
                used_c.add(c)
        return np.array(rows), np.array(cols)

    # ------------------------------------------------------------------
    # Movement analysis
    # ------------------------------------------------------------------

    def _analyze_movement(self, track: Track, frame_width: int):
        """
        Compute speed, approach rate and direction from position history.
        Also determine if this person is blocking the central path corridor.
        """
        hist = list(track.center_history)
        dist_hist = list(track.distance_history)

        # Need at least 2 points
        if len(hist) < 2:
            track.is_stationary = True
            return

        # Use up to `hist_frames` recent frames
        N = min(len(hist), self.hist_frames)
        recent = hist[-N:]
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        pixel_dist = (dx**2 + dy**2) ** 0.5

        # Rough m/s from pixel speed (very approximate without calibration)
        # Assuming 100 pixels ≈ 1 metre at 3 metres distance
        pixel_to_m = 0.01
        track.speed_mps = (pixel_dist * pixel_to_m) / max(N, 1)
        track.is_stationary = track.speed_mps < self.min_speed

        # Approach rate from distance history
        if len(dist_hist) >= 2:
            d_recent = dist_hist[-min(len(dist_hist), self.hist_frames):]
            # Positive delta = moving away, negative = approaching
            track.approach_rate = (d_recent[-1] - d_recent[0]) / max(len(d_recent), 1)
            track.is_approaching = track.approach_rate < self.approach_thr
            track.is_retreating  = track.approach_rate > self.retreat_thr
        else:
            track.approach_rate = 0.0
            track.is_approaching = False
            track.is_retreating  = False

        # Direction in screen space (angle of motion vector)
        if pixel_dist > 5:
            import math
            angle_deg = math.degrees(math.atan2(-dy, dx))  # screen y is flipped
            angle_deg = (angle_deg + 360) % 360
            track.movement_direction = self._angle_to_direction(angle_deg)
        else:
            track.movement_direction = "stationary"

        # Path-blocking analysis
        cx = track.center[0]
        path_left  = frame_width * (0.5 - self.path_center_frac / 2)
        path_right = frame_width * (0.5 + self.path_center_frac / 2)
        track.is_path_blocker = path_left <= cx <= path_right and track.distance_m < 4.0

    @staticmethod
    def _angle_to_direction(angle_deg: float) -> str:
        """Convert movement angle to compass direction label."""
        dirs = [
            (0,    22.5,  "right"),
            (22.5, 67.5,  "lower-right"),
            (67.5, 112.5, "down"),
            (112.5,157.5, "lower-left"),
            (157.5,202.5, "left"),
            (202.5,247.5, "upper-left"),
            (247.5,292.5, "up"),
            (292.5,337.5, "upper-right"),
            (337.5,360,   "right"),
        ]
        for lo, hi, label in dirs:
            if lo <= angle_deg < hi:
                return label
        return "unknown"

    def get_stats(self) -> dict:
        return {
            "active_tracks": len(self._tracks),
            "total_frames": self._frame_num,
        }
