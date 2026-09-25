#!/usr/bin/env python3
"""
train.py
--------
Entrypoint for training the custom Deep Learning Person Detector (Zero-YOLO).

Usage Examples:
    # 1. Train on Penn-Fudan Pedestrian Dataset (auto-downloads if missing):
    python train.py --epochs 5 --batch-size 4

    # 2. Train on GPU if available:
    python train.py --device cuda --epochs 10

    # 3. Custom dataset path:
    python train.py --data path/to/dataset --epochs 10

Author: CrowdAware AI Team
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.download_dataset import download_penn_fudan, generate_synthetic_dataset
from src.training.trainer import PersonDetectionTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Custom PyTorch MobileNetV3 Person Detector (Zero-YOLO)"
    )
    parser.add_argument(
        "--data", "-d", type=str, default="data/PennFudanPed",
        help="Path to dataset root folder containing PNGImages and Annotation"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=4,
        help="Batch size for training (default: 4)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="models",
        help="Directory where trained model checkpoints will be stored (default: models)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "cuda", "mps"],
        help="Compute device (auto-detected if omitted)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)

    print("=" * 65)
    print("  CrowdAware AI - Custom Non-YOLO Model Training Pipeline")
    print("  Architecture: MobileNetV3-Large + SSDLite (Native PyTorch)")
    print("=" * 65)

    # Check if dataset exists, if not auto-download or generate
    if not (data_path / "PNGImages").exists():
        print(f"[Dataset] Target dataset not found at '{data_path}'.")
        parent_dir = data_path.parent
        success = download_penn_fudan(parent_dir)
        if not success:
            print("[Dataset] Auto-download failed. Generating synthetic dataset fallback...")
            generate_synthetic_dataset(parent_dir)

    # Launch trainer
    trainer = PersonDetectionTrainer(
        dataset_path=str(data_path),
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )

    best_checkpoint = trainer.train(epochs=args.epochs)
    print("\n" + "=" * 65)
    print(f"  Training Successfully Finished!")
    print(f"  Trained Model Checkpoint: {best_checkpoint}")
    print(f"  To test live with your trained model, run:")
    print(f"    python main.py --model {best_checkpoint}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
