"""
detector.py
-----------
Person detection module using YOLOv8.
Handles model loading, inference, and result parsing.

Author: CrowdAware AI Team
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """
    Represents a single person detection in a frame.

    Attributes:
        bbox        : Bounding box [x1, y1, x2, y2] in pixels
        confidence  : Detection confidence score [0.0, 1.0]
        class_id    : COCO class ID (0 = person)
        class_name  : Human-readable class name
        center      : (cx, cy) center point of bbox
        area        : Pixel area of bounding box
    """
    bbox: List[int]                      # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    center: Tuple[int, int] = field(init=False)
    area: int = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.area = max(0, (x2 - x1)) * max(0, (y2 - y1))

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def x1(self) -> int: return self.bbox[0]
    @property
    def y1(self) -> int: return self.bbox[1]
    @property
    def x2(self) -> int: return self.bbox[2]
    @property
    def y2(self) -> int: return self.bbox[3]

    def to_tlwh(self) -> List[int]:
        """Convert to top-left width-height format."""
        x1, y1, x2, y2 = self.bbox
        return [x1, y1, x2 - x1, y2 - y1]

    def iou(self, other: "Detection") -> float:
        """Compute Intersection-over-Union with another detection."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def __repr__(self) -> str:
        cx, cy = self.center
        return (f"Detection(class={self.class_name}, conf={self.confidence:.2f}, "
                f"bbox={self.bbox}, center=({cx},{cy}))")


class PersonDetector:
    """
    Real-time person detector using YOLOv8.

    Wraps the Ultralytics YOLO model with:
    - Automatic device selection (CUDA / MPS / CPU)
    - Frame-skip for performance tuning
    - Warm-up inference on load
    - Per-frame timing statistics

    Usage:
        detector = PersonDetector(config)
        detections = detector.detect(frame)
    """

    # COCO class name for person
    PERSON_CLASS_ID = 0
    PERSON_CLASS_NAME = "person"

    def __init__(self, config: dict):
        """
        Initialize the detector.

        Args:
            config: Full application config dict (from settings.yaml)
        """
        self.det_cfg = config.get("detection", {})
        self.model_name = self.det_cfg.get("model", "yolov8n.pt")
        self.conf_threshold = self.det_cfg.get("confidence_threshold", 0.45)
        self.nms_threshold = self.det_cfg.get("nms_threshold", 0.45)
        self.target_classes = self.det_cfg.get("target_classes", [0])
        self.img_size = self.det_cfg.get("img_size", 640)

        # Timing & stats
        self._inference_times: List[float] = []
        self._frame_count = 0
        self._detection_count = 0

        # Select compute device
        self.device = self._select_device(self.det_cfg.get("device", "auto"))
        logger.info(f"[Detector] Using device: {self.device}")

        # Load model
        self.model = self._load_model()
        logger.info(f"[Detector] Model '{self.model_name}' loaded successfully.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_device(self, device_str: str) -> str:
        """Auto-select the best available compute device."""
        if device_str != "auto":
            return device_str
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("[Detector] CUDA GPU detected — using GPU inference.")
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("[Detector] Apple MPS detected — using MPS inference.")
                return "mps"
        except ImportError:
            pass
        logger.info("[Detector] No GPU found — using CPU inference.")
        return "cpu"

    def _load_model(self):
        """Load YOLOv8 model. Downloads weights automatically if needed."""
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_name)
            # Run a single warm-up inference so first real frame isn't slow
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            model(dummy, verbose=False, device=self.device)
            logger.info("[Detector] Warm-up inference complete.")
            return model
        except ImportError as e:
            logger.error("[Detector] ultralytics not installed. Run: pip install ultralytics")
            raise RuntimeError("YOLOv8 not available") from e
        except Exception as e:
            logger.error(f"[Detector] Failed to load model '{self.model_name}': {e}")
            raise

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run person detection on a single BGR frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            List of Detection objects, sorted by confidence descending.
        """
        if frame is None or frame.size == 0:
            logger.warning("[Detector] Received empty frame — skipping.")
            return []

        t0 = time.perf_counter()

        try:
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.nms_threshold,
                classes=self.target_classes,
                imgsz=self.img_size,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"[Detector] Inference error: {e}")
            return []

        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        self._inference_times.append(elapsed_ms)
        if len(self._inference_times) > 60:
            self._inference_times.pop(0)
        self._frame_count += 1

        detections = self._parse_results(results, frame.shape)
        self._detection_count += len(detections)

        logger.debug(f"[Detector] {len(detections)} persons detected in {elapsed_ms:.1f} ms")
        return detections

    def _parse_results(self, results, frame_shape: tuple) -> List[Detection]:
        """Convert raw YOLO results to Detection dataclass list."""
        h, w = frame_shape[:2]
        detections = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                if conf < self.conf_threshold:
                    continue
                if cls_id not in self.target_classes:
                    continue

                # xyxy in pixels
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))

                # Filter tiny detections (noise)
                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue

                cls_name = result.names.get(cls_id, "unknown")
                det = Detection(
                    bbox=[x1, y1, x2, y2],
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                )
                detections.append(det)

        # Sort by confidence, highest first
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def avg_inference_ms(self) -> float:
        """Rolling average inference time in milliseconds."""
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)

    @property
    def current_fps(self) -> float:
        """Estimated detection FPS based on recent inference times."""
        if not self._inference_times or self._inference_times[-1] == 0:
            return 0.0
        return 1000.0 / self._inference_times[-1]

    def get_stats(self) -> dict:
        """Return a dict of detector performance statistics."""
        return {
            "frames_processed": self._frame_count,
            "total_detections": self._detection_count,
            "avg_inference_ms": round(self.avg_inference_ms, 2),
            "current_fps": round(self.current_fps, 1),
            "model": self.model_name,
            "device": self.device,
        }

    def __repr__(self) -> str:
        return (f"PersonDetector(model={self.model_name}, device={self.device}, "
                f"conf={self.conf_threshold})")
