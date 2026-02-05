"""
Satellite Image Dataset Loader

This module provides functionality to load and preprocess satellite imagery
for classification tasks using PyTorch and TorchVision.

Classes (EuroSAT): AnnualCrop, Forest, HerbaceousVegetation, Highway,
    Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path


def get_transforms():
    """
    Create preprocessing transforms compatible with pretrained TorchVision models.

    Returns:
        dict: Dictionary containing 'train' and 'val' transforms.
    """
    # ImageNet normalization statistics (required for pretrained models)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Training transforms: resize, convert to tensor, normalize
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to expected input size
        transforms.ToTensor(),           # Convert PIL Image to tensor [0, 1]
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # Validation transforms: same as training (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return {'train': train_transform, 'val': val_transform}


def load_datasets(data_dir: str):
    """
    Load train and validation datasets using ImageFolder.

    Args:
        data_dir: Path to the root data directory containing 'train' and 'val' folders.

    Returns:
        dict: Dictionary containing 'train' and 'val' datasets.
    """
    data_path = Path(data_dir)
    transforms_dict = get_transforms()

    # Load datasets using ImageFolder (expects class subfolders)
    train_dataset = datasets.ImageFolder(
        root=data_path / 'train',
        transform=transforms_dict['train']
    )

    val_dataset = datasets.ImageFolder(
        root=data_path / 'val',
        transform=transforms_dict['val']
    )

    return {'train': train_dataset, 'val': val_dataset}


def create_dataloaders(datasets_dict: dict, batch_size: int = 32, num_workers: int = 4):
    """
    Create DataLoaders for training and validation.

    Args:
        datasets_dict: Dictionary containing 'train' and 'val' datasets.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses for data loading.

    Returns:
        dict: Dictionary containing 'train' and 'val' DataLoaders.
    """
    train_loader = DataLoader(
        datasets_dict['train'],
        batch_size=batch_size,
        shuffle=True,           # Shuffle training data
        num_workers=num_workers,
        pin_memory=True         # Faster data transfer to GPU
    )

    val_loader = DataLoader(
        datasets_dict['val'],
        batch_size=batch_size,
        shuffle=False,          # No shuffling for validation
        num_workers=num_workers,
        pin_memory=True
    )

    return {'train': train_loader, 'val': val_loader}


def print_dataset_info(datasets_dict: dict, dataloaders_dict: dict):
    """
    Print dataset statistics and sample batch information.

    Args:
        datasets_dict: Dictionary containing datasets.
        dataloaders_dict: Dictionary containing DataLoaders.
    """
    train_dataset = datasets_dict['train']
    train_loader = dataloaders_dict['train']

    # Print class information
    print("=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)

    print(f"\nNumber of classes: {len(train_dataset.classes)}")
    print(f"Class names: {train_dataset.classes}")

    print(f"\nClass-to-index mapping:")
    for class_name, idx in train_dataset.class_to_idx.items():
        print(f"  {class_name}: {idx}")

    # Print dataset sizes
    print(f"\nDataset sizes:")
    print(f"  Training samples: {len(datasets_dict['train'])}")
    print(f"  Validation samples: {len(datasets_dict['val'])}")

    # Get and print sample batch shape
    print("\n" + "=" * 50)
    print("SAMPLE BATCH INFORMATION")
    print("=" * 50)

    # Fetch one batch
    images, labels = next(iter(train_loader))

    print(f"\nBatch image tensor shape: {images.shape}")
    print(f"  - Batch size: {images.shape[0]}")
    print(f"  - Channels (RGB): {images.shape[1]}")
    print(f"  - Height: {images.shape[2]}")
    print(f"  - Width: {images.shape[3]}")

    print(f"\nBatch labels tensor shape: {labels.shape}")
    print(f"Batch labels: {labels.tolist()}")

    # Print tensor statistics (useful for verifying normalization)
    print(f"\nImage tensor statistics (after normalization):")
    print(f"  Min value: {images.min().item():.4f}")
    print(f"  Max value: {images.max().item():.4f}")
    print(f"  Mean value: {images.mean().item():.4f}")


def main():
    """Main function to demonstrate dataset loading."""
    # Configuration
    DATA_DIR = "satellite_data"
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    print("Loading satellite image dataset...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of workers: {NUM_WORKERS}\n")

    # Step 1: Load datasets
    datasets_dict = load_datasets(DATA_DIR)

    # Step 2: Create DataLoaders
    dataloaders_dict = create_dataloaders(
        datasets_dict,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Step 3: Print dataset information and sample batch
    print_dataset_info(datasets_dict, dataloaders_dict)

    print("\n" + "=" * 50)
    print("Dataset loading complete!")
    print("=" * 50)

    return datasets_dict, dataloaders_dict


if __name__ == "__main__":
    main()
