"""
test_mobilenet_training.py
--------------------------
Unit tests for the custom MobileNetV3-SSDLite training pipeline and detector (Zero-YOLO).
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.training.model import create_mobilenet_person_detector, save_checkpoint, load_checkpoint
from src.training.dataset import PennFudanPedDataset, collate_fn
from src.detection.mobilenet_detector import MobileNetPersonDetector


class TestMobileNetArchitecture:

    def test_model_creation(self):
        """Verify model initializes with 2 classes (Background + Person)."""
        model = create_mobilenet_person_detector(num_classes=2, pretrained_backbone=False)
        assert isinstance(model, torch.nn.Module)
        # SSDLite head should have 2 classes
        assert model.head.classification_head.module_list[0][1].out_channels == 12  # 6 anchors * 2 classes

    def test_model_forward_train(self):
        """Verify model returns classification and bbox regression losses during training."""
        model = create_mobilenet_person_detector(num_classes=2, pretrained_backbone=False)
        model.train()

        dummy_img1 = torch.rand(3, 320, 320)
        dummy_img2 = torch.rand(3, 320, 320)
        target1 = {
            "boxes": torch.tensor([[50.0, 50.0, 150.0, 200.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
        target2 = {
            "boxes": torch.tensor([[30.0, 40.0, 120.0, 180.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }

        loss_dict = model([dummy_img1, dummy_img2], [target1, target2])
        assert "bbox_regression" in loss_dict
        assert "classification" in loss_dict
        total_loss = sum(l for l in loss_dict.values())
        assert total_loss.item() > 0

    def test_model_forward_eval(self):
        """Verify model returns predictions in evaluation mode."""
        model = create_mobilenet_person_detector(num_classes=2, pretrained_backbone=False)
        model.eval()

        dummy_img = torch.rand(3, 320, 320)
        with torch.no_grad():
            output = model([dummy_img])

        assert len(output) == 1
        assert "boxes" in output[0]
        assert "scores" in output[0]
        assert "labels" in output[0]


class TestDatasetLoader:

    def test_dataset_loading(self):
        """Verify PennFudanPedDataset can load images and targets if dataset exists."""
        data_path = Path("data/PennFudanPed")
        if not data_path.exists():
            pytest.skip("Dataset not downloaded yet")

        ds = PennFudanPedDataset("data/PennFudanPed")
        assert len(ds) > 0
        img, target = ds[0]

        assert isinstance(img, torch.Tensor)
        assert img.shape[0] == 3  # Channels
        assert "boxes" in target
        assert "labels" in target

    def test_collate_fn(self):
        """Verify collate_fn handles variable detection targets properly."""
        sample_batch = [
            (torch.rand(3, 100, 100), {"boxes": torch.zeros((1, 4))}),
            (torch.rand(3, 100, 100), {"boxes": torch.zeros((2, 4))})
        ]
        imgs, targets = collate_fn(sample_batch)
        assert len(imgs) == 2
        assert len(targets) == 2


class TestMobileNetDetector:

    def test_detector_inference(self):
        """Verify MobileNetPersonDetector runs on a numpy frame and returns Detections."""
        config = {
            "detection": {
                "engine": "mobilenet",
                "model": "models/best_person_detector.pth",
                "confidence_threshold": 0.20,
                "nms_threshold": 0.45,
                "device": "cpu",
                "img_size": 320
            }
        }
        detector = MobileNetPersonDetector(config)
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections = detector.detect(test_frame)
        assert isinstance(detections, list)
        stats = detector.get_stats()
        assert stats["model"] == "MobileNetV3-SSDLite (Zero-YOLO)"
        assert stats["frames_processed"] == 1
