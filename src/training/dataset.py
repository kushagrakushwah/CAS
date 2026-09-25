"""
dataset.py
----------
Custom PyTorch Dataset for loading pedestrian detection data (e.g. Penn-Fudan Pedestrian Dataset).
Parses images and bounding box annotations without requiring any third-party labeling tools.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image


class PennFudanPedDataset(Dataset):
    """
    Penn-Fudan Pedestrian Dataset Loader.

    Each annotation text file contains:
    Bounding box for object N "PennFudanPed" (Xmin, Ymin) - (Xmax, Ymax) : (x1, y1) - (x2, y2)
    """

    def __init__(self, root: str, transforms=None):
        self.root = Path(root)
        self.transforms = transforms

        self.img_dir = self.root / "PNGImages"
        self.ann_dir = self.root / "Annotation"

        if not self.img_dir.exists() or not self.ann_dir.exists():
            raise FileNotFoundError(
                f"Invalid Penn-Fudan dataset directory: {self.root}\n"
                f"Expected subdirectories 'PNGImages' and 'Annotation'.\n"
                f"Please run 'python scripts/download_dataset.py' first."
            )

        # Sort files to ensure alignment
        self.imgs = sorted(list(self.img_dir.glob("*.png")))
        if len(self.imgs) == 0:
            raise ValueError(f"No PNG images found in {self.img_dir}")

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.imgs[idx]
        ann_path = self.ann_dir / f"{img_path.stem}.txt"

        # Load image (RGB)
        img = Image.open(img_path).convert("RGB")

        # Parse bounding boxes from annotation file
        boxes = []
        if ann_path.exists():
            with open(ann_path, "r") as f:
                for line in f:
                    match = re.search(r"\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)", line)
                    if match:
                        x1, y1, x2, y2 = map(float, match.groups())
                        # Ensure valid box dimensions
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])

        if len(boxes) == 0:
            # Fallback for empty/unlabeled samples: small 0-box tensor
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            # Class 1 = Person (Class 0 is background in torchvision detection models)
            labels_tensor = torch.ones((len(boxes),), dtype=torch.int64)
            area_tensor = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            iscrowd_tensor = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
            "area": area_tensor,
            "iscrowd": iscrowd_tensor,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            from torchvision.transforms import functional as F
            img = F.to_tensor(img)

        return img, target


def collate_fn(batch):
    """Custom collate function for object detection batches (variable number of bboxes)."""
    return tuple(zip(*batch))
