"""
distance.py
-----------
Distance estimation from bounding box dimensions.

Uses the pinhole camera model:
    distance = (focal_length_px × known_height_m) / bbox_height_px

The focal length is calibrated once using a known reference (see calibrate.py).

Author: CrowdAware AI Team
"""

import math
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .detector import Detection

logger = logging.getLogger(__name__)


class Zone(str, Enum):
    """
    Proximity zones that map distances to risk levels and colors.
    """
    CRITICAL = "critical"   # < 1.0 m — immediate danger
    CLOSE    = "close"      # 1.0 – 2.5 m
    NEAR     = "near"       # 2.5 – 4.0 m
    MEDIUM   = "medium"     # 4.0 – 7.0 m
    FAR      = "far"        # > 7.0 m


ZONE_LABELS = {
    Zone.CRITICAL: "CRITICAL",
    Zone.CLOSE:    "CLOSE",
    Zone.NEAR:     "NEAR",
    Zone.MEDIUM:   "MEDIUM",
    Zone.FAR:      "FAR",
}


@dataclass
class DistanceEstimate:
    """
    Holds all distance-related information for a single detection.

    Attributes:
        distance_m     : Estimated distance in metres
        zone           : Closest proximity zone
        confidence     : Estimate reliability [0.0, 1.0]
        bbox_height_px : Bounding box height used for estimation
        method         : Estimation method used
    """
    distance_m: float
    zone: Zone
    confidence: float
    bbox_height_px: int
    method: str = "bbox_height"

    @property
    def distance_ft(self) -> float:
        return self.distance_m * 3.28084

    @property
    def zone_label(self) -> str:
        return ZONE_LABELS[self.zone]

    def __repr__(self) -> str:
        return (f"DistanceEstimate(dist={self.distance_m:.2f}m, "
                f"zone={self.zone.value}, conf={self.confidence:.2f})")


class DistanceEstimator:
    """
    Estimates real-world distance of detected persons using bounding-box height.

    The approach is based on the thin-lens (pinhole camera) equation:
        distance = (F × H_real) / H_px

    where:
        F       = focal length in pixels (calibrated)
        H_real  = average adult height in metres (default 1.70 m)
        H_px    = detected person bounding-box height in pixels

    Limitations:
        - Works best for full-body detections (head to toe visible)
        - Assumes a standing person of average height
        - Error increases for partial occlusions and unusual poses

    Usage:
        estimator = DistanceEstimator(config)
        estimate  = estimator.estimate(detection, frame_height)
    """

    def __init__(self, config: dict):
        dist_cfg = config.get("distance", {})
        zone_cfg = dist_cfg.get("zones", {})

        self.focal_length_px     = float(dist_cfg.get("focal_length_px", 615.0))
        self.known_height_m      = float(dist_cfg.get("known_person_height_m", 1.70))

        # Zone thresholds (metres)
        self.zone_critical = float(zone_cfg.get("critical", 1.0))
        self.zone_close    = float(zone_cfg.get("close",    2.5))
        self.zone_near     = float(zone_cfg.get("near",     4.0))
        self.zone_medium   = float(zone_cfg.get("medium",   7.0))

        logger.info(
            f"[Distance] Focal={self.focal_length_px:.1f}px, "
            f"PersonHeight={self.known_height_m}m"
        )
        logger.info(
            f"[Distance] Zones — critical<{self.zone_critical}m, "
            f"close<{self.zone_close}m, near<{self.zone_near}m, "
            f"medium<{self.zone_medium}m"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def estimate(self, detection: Detection, frame_height: int) -> DistanceEstimate:
        """
        Estimate distance for a single Detection.

        Args:
            detection    : A Detection object with valid bbox
            frame_height : Height of the source frame in pixels

        Returns:
            DistanceEstimate with distance, zone and confidence
        """
        bbox_h = detection.height

        # Clamp to avoid division by zero / nonsensical values
        bbox_h = max(bbox_h, 1)

        distance_m = self._pinhole_distance(bbox_h)

        # Confidence is higher when person is fully visible and bbox is tall
        # Heuristic: full-body detection ~80% of frame height is ideal
        ideal_ratio = 0.70
        actual_ratio = bbox_h / max(frame_height, 1)
        conf = min(1.0, actual_ratio / ideal_ratio)

        # Partial detections (very small bbox) are less reliable
        if bbox_h < 60:
            conf *= 0.6
        elif bbox_h < 120:
            conf *= 0.8

        zone = self._classify_zone(distance_m)

        return DistanceEstimate(
            distance_m=round(distance_m, 2),
            zone=zone,
            confidence=round(conf, 2),
            bbox_height_px=bbox_h,
            method="bbox_height",
        )

    def estimate_from_height_px(self, height_px: int) -> float:
        """
        Estimate distance from a bounding box height in pixels.
        Useful for quick estimations without a Detection object.
        """
        return self._pinhole_distance(max(height_px, 1))

    def calibrate(
        self,
        known_distance_m: float,
        measured_height_px: int,
    ) -> float:
        """
        Compute the focal length for a known reference measurement.

        Call this with a person standing at a known distance and record their
        bounding-box height. Returns the new focal_length_px value.

        Args:
            known_distance_m   : True distance of the person (metres)
            measured_height_px : Measured bbox height in pixels

        Returns:
            Computed focal length in pixels — update config with this value.
        """
        focal = (known_distance_m * measured_height_px) / self.known_height_m
        logger.info(
            f"[Distance] Calibration result: focal_length_px = {focal:.1f} "
            f"(measured at {known_distance_m}m, height={measured_height_px}px)"
        )
        return focal

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pinhole_distance(self, bbox_h_px: int) -> float:
        """Core pinhole camera distance formula."""
        return (self.focal_length_px * self.known_height_m) / bbox_h_px

    def _classify_zone(self, distance_m: float) -> Zone:
        """Map a distance in metres to a Zone enum value."""
        if distance_m <= self.zone_critical:
            return Zone.CRITICAL
        elif distance_m <= self.zone_close:
            return Zone.CLOSE
        elif distance_m <= self.zone_near:
            return Zone.NEAR
        elif distance_m <= self.zone_medium:
            return Zone.MEDIUM
        else:
            return Zone.FAR

    def get_zone_color_bgr(self, zone: Zone) -> Tuple[int, int, int]:
        """Return BGR color tuple for a given zone."""
        colors = {
            Zone.CRITICAL : (0,   0,   255),   # Red
            Zone.CLOSE    : (0,   100, 255),   # Orange
            Zone.NEAR     : (0,   220, 220),   # Yellow
            Zone.MEDIUM   : (0,   200, 0),     # Green
            Zone.FAR      : (200, 150, 0),     # Teal-blue
        }
        return colors.get(zone, (200, 200, 200))
