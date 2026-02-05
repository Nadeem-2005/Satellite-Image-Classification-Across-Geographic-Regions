"""
Prepare EuroSAT Dataset — Train/Val Split

Splits the raw EuroSAT_RGB dataset (flat class folders) into a train/val
directory structure expected by torchvision.datasets.ImageFolder.

Source:  Datasets/EuroSAT_RGB/{class}/*.jpg
Output:  satellite_data/train/{class}/*.jpg
         satellite_data/val/{class}/*.jpg

Split ratio: 80% train, 20% validation
"""

import os
import shutil
import random
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────
SOURCE_DIR = Path("Datasets/EuroSAT_RGB")
OUTPUT_DIR = Path("satellite_data")
TRAIN_RATIO = 0.8
SEED = 42


def split_dataset(source_dir: Path, output_dir: Path, train_ratio: float, seed: int):
    """
    Split each class folder into train and val subsets.

    Args:
        source_dir: Path to the raw EuroSAT_RGB directory.
        output_dir: Path to create the train/val directory structure.
        train_ratio: Fraction of images to use for training.
        seed: Random seed for reproducible splits.
    """
    random.seed(seed)

    # Get all class directories (skip hidden files)
    class_dirs = sorted([
        d for d in source_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not class_dirs:
        raise FileNotFoundError(f"No class folders found in {source_dir}")

    print(f"Source directory : {source_dir}")
    print(f"Output directory : {output_dir}")
    print(f"Train ratio      : {train_ratio}")
    print(f"Seed             : {seed}")
    print(f"Classes found    : {len(class_dirs)}")
    print()

    total_train = 0
    total_val = 0

    for class_dir in class_dirs:
        class_name = class_dir.name

        # Collect all image files in this class
        images = sorted([
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif")
        ])

        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Create output directories
        train_dir = output_dir / "train" / class_name
        val_dir = output_dir / "val" / class_name
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        # Copy files
        for img in train_images:
            shutil.copy2(img, train_dir / img.name)
        for img in val_images:
            shutil.copy2(img, val_dir / img.name)

        total_train += len(train_images)
        total_val += len(val_images)

        print(f"  {class_name:30s}  train: {len(train_images):5d}  val: {len(val_images):5d}")

    print()
    print(f"Total  train: {total_train}  val: {total_val}")
    print(f"Dataset prepared at: {output_dir.resolve()}")


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Source directory not found: {SOURCE_DIR}\n"
            "Make sure the EuroSAT_RGB dataset is at Datasets/EuroSAT_RGB/"
        )

    # Warn before overwriting
    if OUTPUT_DIR.exists():
        print(f"Output directory {OUTPUT_DIR} already exists. Removing it for a clean split.")
        shutil.rmtree(OUTPUT_DIR)

    split_dataset(SOURCE_DIR, OUTPUT_DIR, TRAIN_RATIO, SEED)


if __name__ == "__main__":
    main()
