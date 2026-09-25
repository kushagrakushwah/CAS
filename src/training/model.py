"""
model.py
--------
Custom Deep Learning Object Detection Architecture using MobileNetV3 + SSDLite (Zero-YOLO).
Provides:
- Lightweight, high-throughput detector for real-time edge CPU/GPU execution.
- Head replacement for 2 classes: Class 0 (Background) and Class 1 (Person).
- Checkpoint loading and saving helpers.
"""

from typing import Optional
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.ssdlite import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
    SSDLiteHead
)


def create_mobilenet_person_detector(
    num_classes: int = 2,
    pretrained_backbone: bool = True
) -> nn.Module:
    """
    Creates an SSDLite320 model with a MobileNetV3-Large backbone for person detection (2 classes).
    """
    weights_backbone = "DEFAULT" if pretrained_backbone else None
    model = ssdlite320_mobilenet_v3_large(
        num_classes=num_classes,
        weights_backbone=weights_backbone
    )
    return model


def save_checkpoint(model: nn.Module, optimizer, epoch: int, loss: float, filepath: str):
    """Save model checkpoint dictionary."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "loss": loss,
        "architecture": "ssdlite320_mobilenet_v3_large",
        "num_classes": 2,
    }, filepath)
    print(f"[Model] Saved checkpoint to: {filepath}")


def load_checkpoint(filepath: str, device: str = "cpu") -> nn.Module:
    """Load model from checkpoint file."""
    model = create_mobilenet_person_detector(num_classes=2, pretrained_backbone=True)
    checkpoint = torch.load(filepath, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model
