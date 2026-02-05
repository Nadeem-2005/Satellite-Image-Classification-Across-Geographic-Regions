"""
Transfer Learning with ResNet-50 for Satellite Image Classification

This module sets up a pretrained ResNet-50 as a fixed feature extractor,
replacing only the final classification head for 10 EuroSAT land-use classes:
AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
Pasture, PermanentCrop, Residential, River, SeaLake.

Transfer learning rationale:
- ResNet-50 pretrained on ImageNet has learned rich, general-purpose visual
  features (edges, textures, shapes) in its convolutional backbone.
- Satellite RGB imagery shares enough low-level visual structure with natural
  images that these features transfer well.
- By freezing the backbone and training only the classifier head, we avoid
  overfitting on small datasets and reduce compute requirements significantly.
"""

import torch
import torch.nn as nn
from torchvision import models
from data_loader import load_datasets, create_dataloaders

NUM_CLASSES = 10


def create_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Load a pretrained ResNet-50 and replace its classifier head.

    The backbone (all convolutional + batch norm layers) is frozen so that
    backpropagation only updates the new fully connected layer. This turns
    the network into a fixed feature extractor with a trainable linear probe.

    Args:
        num_classes: Number of output classes.

    Returns:
        Modified ResNet-50 model.
    """
    # Load ResNet-50 with pretrained ImageNet weights
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze every parameter in the backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer (originally 1000-class ImageNet output)
    # with a new layer for our 10 satellite land-use classes.
    # nn.Linear parameters have requires_grad=True by default, so only this layer trains.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Run one full training epoch.

    Args:
        model: The neural network.
        dataloader: Training DataLoader.
        optimizer: Optimizer (updates only unfrozen parameters).
        criterion: Loss function.
        device: Target device (cpu or cuda).

    Returns:
        dict with 'loss' (average) and 'accuracy' (percentage).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return {"loss": avg_loss, "accuracy": accuracy}


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Evaluate the model on the validation set.

    Args:
        model: The neural network.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Target device (cpu or cuda).

    Returns:
        dict with 'loss' (average) and 'accuracy' (percentage).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return {"loss": avg_loss, "accuracy": accuracy}


def main():
    """Run one training epoch to verify the pipeline end-to-end."""

    # ── Configuration ──────────────────────────────────────────────
    DATA_DIR = "satellite_data"
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    LEARNING_RATE = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Data ───────────────────────────────────────────────────────
    datasets_dict = load_datasets(DATA_DIR)
    loaders = create_dataloaders(datasets_dict, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # ── Model ──────────────────────────────────────────────────────
    model = create_model(num_classes=NUM_CLASSES)
    model.to(device)

    # Verify that only the classifier head is trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Device           : {device}")
    print(f"Trainable params : {trainable:,}")
    print(f"Frozen params    : {frozen:,}")

    # ── Loss & Optimizer ───────────────────────────────────────────
    # CrossEntropyLoss is standard for multi-class single-label classification.
    criterion = nn.CrossEntropyLoss()

    # Optimise only the parameters that require gradients (the new fc layer).
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    # ── Train for one epoch ────────────────────────────────────────
    print("\n── Epoch 1 ──────────────────────────────────────────")
    train_metrics = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
    val_metrics = validate(model, loaders["val"], criterion, device)

    print(f"Train Loss : {train_metrics['loss']:.4f}  |  Train Acc : {train_metrics['accuracy']:.2f}%")
    print(f"Val   Loss : {val_metrics['loss']:.4f}  |  Val   Acc : {val_metrics['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
