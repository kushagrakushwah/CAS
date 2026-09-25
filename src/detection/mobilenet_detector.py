"""
mobilenet_detector.py
---------------------
Independent Person Detector using custom-trained MobileNetV3-SSDLite (Zero-YOLO).
Features:
- Pure PyTorch / Torchvision implementation (no Ultralytics, no Darknet).
- Automatic device fallback (CUDA / Apple MPS / CPU).
- High inference speed (320x320 optimized multi-scale feature pyramids).
- Compatible with Detection dataclass used by pipeline, tracker, and distance estimator.
"""

import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
import torch
import torchvision

from .detector import Detection
from ..training.model import load_checkpoint, create_mobilenet_person_detector

logger = logging.getLogger(__name__)


class MobileNetPersonDetector:
    """
    Real-time Person Detector powered by MobileNetV3 + SSDLite.
    Replaces YOLO with our custom-trained PyTorch architecture.
    """

    PERSON_CLASS_ID = 1  # 0 is background, 1 is person
    PERSON_CLASS_NAME = "person"

    def __init__(self, config: dict, checkpoint_path: Optional[str] = None):
        self.det_cfg = config.get("detection", {})

        # Checkpoint priority: explicit param > config > default search paths
        if checkpoint_path is None:
            checkpoint_path = self.det_cfg.get("model", "models/best_person_detector.pth")

        self.checkpoint_path = checkpoint_path
        self.conf_threshold = float(self.det_cfg.get("confidence_threshold", 0.40))
        self.nms_threshold = float(self.det_cfg.get("nms_threshold", 0.45))
        self.img_size = int(self.det_cfg.get("img_size", 320))

        # Device selection
        self.device = self._select_device(self.det_cfg.get("device", "auto"))
        logger.info(f"[MobileNetDetector] Selected compute device: {self.device}")

        # Timing stats
        self._inference_times: List[float] = []
        self._frame_count = 0
        self._detection_count = 0

        # Load model weights
        self.model = self._load_model()
        logger.info(f"[MobileNetDetector] Model loaded from: {self.checkpoint_path}")

    def _select_device(self, device_str: str) -> torch.device:
        if device_str != "auto":
            return torch.device(device_str)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self) -> torch.nn.Module:
        """Load trained weights, or initialize default model if checkpoint not yet trained."""
        ckpt_file = Path(self.checkpoint_path)

        if ckpt_file.exists():
            logger.info(f"[MobileNetDetector] Loading custom checkpoint: {ckpt_file}")
            model = load_checkpoint(str(ckpt_file), device=str(self.device))
        else:
            logger.warning(
                f"[MobileNetDetector] Checkpoint '{ckpt_file}' not found.\n"
                f"Initializing pre-trained backbone. Run 'python train.py' to train on dataset."
            )
            model = create_mobilenet_person_detector(num_classes=2, pretrained_backbone=True)
            model.to(self.device)
            model.eval()

        # Run warm-up inference
        try:
            dummy = [torch.zeros((3, self.img_size, self.img_size), device=self.device)]
            with torch.no_grad():
                model(dummy)
            logger.info("[MobileNetDetector] Warm-up inference complete.")
        except Exception as e:
            logger.debug(f"[MobileNetDetector] Warmup pass notice: {e}")

        return model

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run person detection on an input BGR frame.

        Args:
            frame: OpenCV BGR image (H, W, 3)

        Returns:
            List of Detection dataclass objects.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        t0 = time.perf_counter()

        # Preprocessing: BGR -> RGB -> Float32 [0.0, 1.0] -> Tensor (C, H, W)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor_img = tensor_img.to(self.device)

        with torch.no_grad():
            predictions = self.model([tensor_img])[0]

        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        self._inference_times.append(elapsed_ms)
        if len(self._inference_times) > 60:
            self._inference_times.pop(0)
        self._frame_count += 1

        # Parse outputs
        boxes = predictions["boxes"].cpu()
        scores = predictions["scores"].cpu()
        labels = predictions["labels"].cpu()

        # Filter by class 1 (person) and confidence threshold
        mask = (labels == self.PERSON_CLASS_ID) & (scores >= self.conf_threshold)
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]

        if len(filtered_boxes) == 0:
            return []

        # Non-Maximum Suppression (NMS)
        keep = torchvision.ops.nms(filtered_boxes, filtered_scores, self.nms_threshold)
        nms_boxes = filtered_boxes[keep].numpy()
        nms_scores = filtered_scores[keep].numpy()

        detections = []
        for box, score in zip(nms_boxes, nms_scores):
            x1, y1, x2, y2 = box.astype(int)
            # Clamp to frame bounds
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))

            # Discard tiny noise
            if (x2 - x1) < 15 or (y2 - y1) < 25:
                continue

            det = Detection(
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                confidence=float(score),
                class_id=0,  # Pipeline standard person class ID
                class_name="person",
            )
            detections.append(det)

        detections.sort(key=lambda d: d.confidence, reverse=True)
        self._detection_count += len(detections)
        return detections

    @property
    def avg_inference_ms(self) -> float:
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)

    @property
    def current_fps(self) -> float:
        if not self._inference_times or self._inference_times[-1] == 0:
            return 0.0
        return 1000.0 / self._inference_times[-1]

    def get_stats(self) -> dict:
        return {
            "frames_processed": self._frame_count,
            "total_detections": self._detection_count,
            "avg_inference_ms": round(self.avg_inference_ms, 2),
            "current_fps": round(self.current_fps, 1),
            "model": "MobileNetV3-SSDLite (Zero-YOLO)",
            "device": str(self.device),
        }
