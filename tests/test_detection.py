"""
test_detection.py
-----------------
Unit tests for the detection, distance estimation, and tracking modules.

Run with: pytest tests/ -v --cov=src
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def sample_config():
    """Minimal config dict for testing."""
    return {
        "detection": {
            "model": "yolov8n.pt",
            "confidence_threshold": 0.45,
            "nms_threshold": 0.45,
            "target_classes": [0],
            "device": "cpu",
            "img_size": 640,
        },
        "distance": {
            "focal_length_px": 615.0,
            "known_person_height_m": 1.70,
            "zones": {
                "critical": 1.0,
                "close": 2.5,
                "near": 4.0,
                "medium": 7.0,
                "far": 999.0,
            },
        },
        "tracking": {
            "enabled": True,
            "max_age": 30,
            "min_hits": 1,       # Low for testing
            "iou_threshold": 0.3,
        },
        "movement": {
            "history_frames": 15,
            "min_speed_threshold": 0.05,
            "approach_threshold": -0.15,
            "retreat_threshold": 0.15,
        },
        "audio": {
            "enabled": False,    # Disabled for testing
            "engine": "pyttsx3",
            "volume": 0.9,
            "rate": 175,
            "spatial_audio": True,
            "beep_alerts": False,
            "cooldowns": {
                "critical": 0.0,  # Zero cooldown for testing
                "close": 0.0,
                "near": 0.0,
                "path_blocked": 0.0,
                "crowd": 0.0,
                "approach": 0.0,
            },
            "messages": {
                "critical": "Warning! Person {distance:.1f} metres ahead",
                "close": "Person very close, {distance:.1f} metres",
                "near": "Person nearby at {distance:.1f} metres",
                "approaching": "Someone approaching from your {direction}",
                "path_blocked": "Path ahead is blocked",
                "crowd_alert": "{count} people detected nearby",
                "clear": "Path clear",
            },
        },
        "zones": {
            "path_center_fraction": 0.40,
            "path_block_threshold": 0.35,
            "crowd_threshold": 5,
            "high_density_threshold": 10,
        },
    }


@pytest.fixture
def sample_frame():
    """A black 720p test frame."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


# ------------------------------------------------------------------
# Detection tests
# ------------------------------------------------------------------

class TestDetection:

    def test_detection_creation(self):
        from src.detection.detector import Detection
        det = Detection(
            bbox=[100, 50, 250, 400],
            confidence=0.87,
            class_id=0,
            class_name="person",
        )
        assert det.width  == 150
        assert det.height == 350
        assert det.center == (175, 225)
        assert det.area   == 150 * 350

    def test_detection_tlwh(self):
        from src.detection.detector import Detection
        det = Detection(bbox=[10, 20, 110, 220], confidence=0.9, class_id=0, class_name="person")
        tlwh = det.to_tlwh()
        assert tlwh == [10, 20, 100, 200]

    def test_detection_iou_identical(self):
        from src.detection.detector import Detection
        d = Detection(bbox=[0, 0, 100, 100], confidence=0.9, class_id=0, class_name="person")
        assert d.iou(d) == pytest.approx(1.0)

    def test_detection_iou_no_overlap(self):
        from src.detection.detector import Detection
        d1 = Detection(bbox=[0,   0, 50, 50],   confidence=0.9, class_id=0, class_name="person")
        d2 = Detection(bbox=[100, 100, 200, 200], confidence=0.9, class_id=0, class_name="person")
        assert d1.iou(d2) == pytest.approx(0.0)

    def test_detection_iou_partial(self):
        from src.detection.detector import Detection
        d1 = Detection(bbox=[0,  0, 100, 100], confidence=0.9, class_id=0, class_name="person")
        d2 = Detection(bbox=[50, 0, 150, 100], confidence=0.9, class_id=0, class_name="person")
        # 50×100 = 5000 intersect, union = 10000+10000-5000=15000
        assert d1.iou(d2) == pytest.approx(5000 / 15000, rel=1e-4)

    def test_detection_sort_by_confidence(self):
        from src.detection.detector import Detection
        dets = [
            Detection(bbox=[0,0,10,10], confidence=0.5, class_id=0, class_name="person"),
            Detection(bbox=[0,0,10,10], confidence=0.9, class_id=0, class_name="person"),
            Detection(bbox=[0,0,10,10], confidence=0.7, class_id=0, class_name="person"),
        ]
        dets.sort(key=lambda d: d.confidence, reverse=True)
        assert dets[0].confidence == 0.9
        assert dets[-1].confidence == 0.5


# ------------------------------------------------------------------
# Distance estimation tests
# ------------------------------------------------------------------

class TestDistanceEstimation:

    def test_zone_classification_critical(self, sample_config):
        from src.detection.distance import DistanceEstimator, Zone
        est = DistanceEstimator(sample_config)
        assert est._classify_zone(0.5)  == Zone.CRITICAL
        assert est._classify_zone(1.0)  == Zone.CRITICAL

    def test_zone_classification_close(self, sample_config):
        from src.detection.distance import DistanceEstimator, Zone
        est = DistanceEstimator(sample_config)
        assert est._classify_zone(1.5)  == Zone.CLOSE
        assert est._classify_zone(2.5)  == Zone.CLOSE

    def test_zone_classification_near(self, sample_config):
        from src.detection.distance import DistanceEstimator, Zone
        est = DistanceEstimator(sample_config)
        assert est._classify_zone(3.0)  == Zone.NEAR
        assert est._classify_zone(4.0)  == Zone.NEAR

    def test_zone_classification_medium(self, sample_config):
        from src.detection.distance import DistanceEstimator, Zone
        est = DistanceEstimator(sample_config)
        assert est._classify_zone(5.0)  == Zone.MEDIUM
        assert est._classify_zone(7.0)  == Zone.MEDIUM

    def test_zone_classification_far(self, sample_config):
        from src.detection.distance import DistanceEstimator, Zone
        est = DistanceEstimator(sample_config)
        assert est._classify_zone(10.0) == Zone.FAR

    def test_pinhole_distance_formula(self, sample_config):
        """Verify the distance formula: d = F*H_real / H_px"""
        from src.detection.distance import DistanceEstimator
        est = DistanceEstimator(sample_config)
        # With focal=615, person_height=1.70m, bbox_h=?
        # At 2.0m: H_px = (615 * 1.70) / 2.0 = 522.75 → ~523 px
        h_px = 523
        d = est.estimate_from_height_px(h_px)
        assert abs(d - 2.0) < 0.1

    def test_calibration_returns_focal(self, sample_config):
        from src.detection.distance import DistanceEstimator
        est = DistanceEstimator(sample_config)
        # If person is 2m away and bbox is 523px tall, focal should be ~615
        fl = est.calibrate(known_distance_m=2.0, measured_height_px=523)
        expected = (2.0 * 523) / 1.70
        assert abs(fl - expected) < 1.0

    def test_estimate_returns_distance_estimate(self, sample_config):
        from src.detection.detector import Detection
        from src.detection.distance import DistanceEstimator, DistanceEstimate
        est = DistanceEstimator(sample_config)
        det = Detection(bbox=[100, 50, 300, 573], confidence=0.9, class_id=0, class_name="person")
        result = est.estimate(det, frame_height=720)
        assert isinstance(result, DistanceEstimate)
        assert result.distance_m > 0
        assert 0.0 <= result.confidence <= 1.0

    def test_estimate_closer_person_has_larger_bbox(self, sample_config):
        from src.detection.detector import Detection
        from src.detection.distance import DistanceEstimator
        est = DistanceEstimator(sample_config)

        # Tall bbox = close person
        close_det = Detection(bbox=[100, 0, 200, 600], confidence=0.9, class_id=0, class_name="person")
        # Short bbox = far person
        far_det   = Detection(bbox=[100, 300, 200, 420], confidence=0.9, class_id=0, class_name="person")

        close_est = est.estimate(close_det, frame_height=720)
        far_est   = est.estimate(far_det,   frame_height=720)

        assert close_est.distance_m < far_est.distance_m


# ------------------------------------------------------------------
# Tracker tests
# ------------------------------------------------------------------

class TestTracker:

    def test_new_detection_creates_track(self, sample_config):
        from src.detection.detector import Detection
        from src.detection.tracker import MultiPersonTracker
        tracker = MultiPersonTracker(sample_config)
        det = Detection(bbox=[100, 50, 300, 500], confidence=0.9, class_id=0, class_name="person")
        tracks = tracker.update([det], [None], (720, 1280))
        assert len(tracks) >= 1

    def test_no_detections_returns_empty(self, sample_config):
        from src.detection.tracker import MultiPersonTracker
        tracker = MultiPersonTracker(sample_config)
        tracks = tracker.update([], [], (720, 1280))
        assert len(tracks) == 0

    def test_track_id_persists_across_frames(self, sample_config):
        from src.detection.detector import Detection
        from src.detection.tracker import MultiPersonTracker
        tracker = MultiPersonTracker(sample_config)

        det1 = Detection(bbox=[100, 50, 200, 400], confidence=0.9, class_id=0, class_name="person")
        det2 = Detection(bbox=[102, 52, 202, 402], confidence=0.9, class_id=0, class_name="person")  # slight movement

        tracks1 = tracker.update([det1], [None], (720, 1280))
        tracks2 = tracker.update([det2], [None], (720, 1280))

        if tracks1 and tracks2:
            assert tracks1[0].track_id == tracks2[0].track_id

    def test_multiple_detections_multiple_tracks(self, sample_config):
        from src.detection.detector import Detection
        from src.detection.tracker import MultiPersonTracker
        tracker = MultiPersonTracker(sample_config)

        dets = [
            Detection(bbox=[50,  100, 150, 500], confidence=0.9, class_id=0, class_name="person"),
            Detection(bbox=[600, 100, 700, 500], confidence=0.8, class_id=0, class_name="person"),
        ]
        # Update multiple times to confirm tracks
        for _ in range(3):
            tracks = tracker.update(dets, [None, None], (720, 1280))

        assert len(tracks) == 2
        ids = {t.track_id for t in tracks}
        assert len(ids) == 2   # Different IDs


# ------------------------------------------------------------------
# Alert engine tests
# ------------------------------------------------------------------

class TestAlertEngine:

    def test_alert_engine_disabled(self, sample_config):
        """Disabled engine should not raise errors."""
        from src.audio.alert_engine import AlertEngine
        engine = AlertEngine(sample_config)
        assert not engine.enabled
        engine.start()
        engine.alert_critical("Test")   # Should be a no-op
        engine.stop()

    def test_compute_pan_center(self, sample_config):
        from src.audio.alert_engine import AlertEngine
        engine = AlertEngine(sample_config)
        pan = engine.compute_pan(640, 1280)
        assert abs(pan) < 0.01   # Center → ~0.0

    def test_compute_pan_left(self, sample_config):
        from src.audio.alert_engine import AlertEngine
        engine = AlertEngine(sample_config)
        pan = engine.compute_pan(0, 1280)
        assert pan == pytest.approx(-1.0)   # Far left

    def test_compute_pan_right(self, sample_config):
        from src.audio.alert_engine import AlertEngine
        engine = AlertEngine(sample_config)
        pan = engine.compute_pan(1280, 1280)
        assert pan == pytest.approx(1.0)    # Far right

    def test_stats_structure(self, sample_config):
        from src.audio.alert_engine import AlertEngine
        engine = AlertEngine(sample_config)
        stats = engine.get_stats()
        assert "alerts_issued" in stats
        assert "alerts_suppressed" in stats
        assert "engine" in stats


# ------------------------------------------------------------------
# Calibration tests
# ------------------------------------------------------------------

class TestCalibration:

    def test_focal_length_formula(self):
        from scripts.calibrate import compute_focal_length
        # At 2m, 523px bbox height, 1.70m person → focal ≈ 615
        fl = compute_focal_length(2.0, 523, 1.70)
        expected = (2.0 * 523) / 1.70
        assert abs(fl - expected) < 0.1

    def test_focal_length_zero_height_raises(self):
        from scripts.calibrate import compute_focal_length
        with pytest.raises((ValueError, ZeroDivisionError)):
            compute_focal_length(2.0, 0, 1.70)

    def test_focal_length_zero_distance_raises(self):
        from scripts.calibrate import compute_focal_length
        with pytest.raises(ValueError):
            compute_focal_length(0.0, 300, 1.70)

    def test_distance_verification_table(self):
        """Verify that focal length correctly round-trips."""
        from scripts.calibrate import compute_focal_length
        fl   = compute_focal_length(3.0, 348, 1.70)
        # Reconstruct distance from focal
        dist = (fl * 1.70) / 348
        assert abs(dist - 3.0) < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
