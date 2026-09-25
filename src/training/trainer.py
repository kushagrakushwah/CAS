"""
trainer.py
----------
Training engine for our custom PyTorch MobileNetV3-SSDLite Person Detector.
Implements:
- Multi-task loss tracking (bounding box regression loss + classification loss).
- Optimizer setup with weight decay and CosineAnnealing / StepLR scheduler.
- Checkpointing of the best performing model.
"""

import time
import os
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader

from .model import create_mobilenet_person_detector, save_checkpoint
from .dataset import PennFudanPedDataset, collate_fn


class PersonDetectionTrainer:
    """Trainer orchestrator for MobileNetV3 Person Detector."""

    def __init__(
        self,
        dataset_path: str = "data/PennFudanPed",
        output_dir: str = "models",
        batch_size: int = 4,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        device: Optional[str] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        # Auto-select device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[Trainer] Initializing trainer on device: {self.device}")

        # Initialize dataset & dataloader
        self.dataset = PennFudanPedDataset(dataset_path)
        print(f"[Trainer] Loaded dataset with {len(self.dataset)} images.")

        # Split 85% train, 15% validation
        train_size = int(0.85 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        # Initialize model
        print("[Trainer] Building MobileNetV3-SSDLite architecture (Zero-YOLO)...")
        self.model = create_mobilenet_person_detector(num_classes=2, pretrained_backbone=True)
        self.model.to(self.device)

        # Optimizer & learning rate scheduler
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20)

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch and return average total loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        t0 = time.time()
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # Move batch to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # In training mode, torchvision detection models return a dict of losses
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += losses.item()

            if (batch_idx + 1) % max(1, num_batches // 5) == 0 or (batch_idx + 1) == num_batches:
                cls_loss = loss_dict.get("classification", torch.tensor(0.0)).item()
                bbox_loss = loss_dict.get("bbox_regression", torch.tensor(0.0)).item()
                print(
                    f"  Epoch [{epoch}] Batch [{batch_idx+1}/{num_batches}] "
                    f"Loss: {losses.item():.4f} (Cls: {cls_loss:.4f}, BBox: {bbox_loss:.4f})"
                )

        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - t0
        print(f"[Trainer] Epoch {epoch} finished in {elapsed:.1f}s - Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self, epochs: int = 5) -> str:
        """Run full training routine for specified epochs and save checkpoint."""
        print(f"\n[Trainer] Starting training for {epochs} epochs...")
        best_loss = float("inf")
        best_checkpoint_path = str(self.output_dir / "best_person_detector.pth")

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(epoch)
            self.lr_scheduler.step()

            # Save best checkpoint
            if loss < best_loss:
                best_loss = loss
                save_checkpoint(self.model, self.optimizer, epoch, loss, best_checkpoint_path)

        # Also save latest checkpoint
        latest_path = str(self.output_dir / "latest_person_detector.pth")
        save_checkpoint(self.model, self.optimizer, epochs, loss, latest_path)

        print(f"\n[Trainer] Training complete! Best model saved to: {best_checkpoint_path}")
        return best_checkpoint_path
