"""
Federated Training Orchestration (Optimized with Feature Caching)

Key optimization: Backbone features are extracted ONCE and cached as tensors.
All training (federated and centralized) operates on cached 2048-dim feature
vectors instead of re-running the ResNet-50 backbone each epoch.
This yields ~50-100x speedup on the training loop.

Coordinates the full federated learning loop:
  1. Extract backbone features once (one-time cost)
  2. Partition cached features across clients
  3. For each round: broadcast → local FC train → aggregate → evaluate
  4. Log per-round global validation accuracy and per-client metrics
"""

import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from data_loader import load_datasets, create_dataloaders
from model import (
    extract_features,
    FCClassifier,
    train_fc_epoch,
    validate_fc,
)
from fed_data import (
    partition_iid,
    partition_non_iid,
    partition_non_iid_dirichlet,
    create_client_feature_dataloaders,
    print_partition_stats,
)
from fed_client import FederatedClient
from fed_server import FederatedServer


# ── Feature Extraction Helper ────────────────────────────────────────────

def extract_and_cache(data_dir, device, num_workers=2, batch_size=128):
    """
    Extract backbone features for train and val sets. Call once, reuse everywhere.

    Returns:
        dict with train_features, train_labels, val_features, val_labels,
        train_dataset, and class_names.
    """
    datasets_dict = load_datasets(data_dir)
    train_dataset = datasets_dict["train"]
    class_names = train_dataset.classes

    print("Extracting backbone features (one-time cost)...")
    t0 = time.time()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    train_features, train_labels = extract_features(train_loader, device)

    val_loader = DataLoader(
        datasets_dict["val"], batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    val_features, val_labels = extract_features(val_loader, device)

    elapsed = time.time() - t0
    print(f"  Train features: {train_features.shape} | Val features: {val_features.shape}")
    print(f"  Extraction time: {elapsed:.1f}s")

    return {
        "train_features": train_features,
        "train_labels": train_labels,
        "val_features": val_features,
        "val_labels": val_labels,
        "train_dataset": train_dataset,
        "class_names": class_names,
    }


# ── Federated Training ───────────────────────────────────────────────────

def run_federated(
    num_clients=5,
    num_rounds=20,
    local_epochs=5,
    partition="iid",
    alpha=0.5,
    classes_per_client=2,
    batch_size=512,
    lr=1e-3,
    num_workers=2,
    data_dir="satellite_data",
    num_classes=10,
    device=None,
    seed=42,
    cached_data=None,
):
    """
    Run the full FedAvg federated training loop with feature caching.

    Pass cached_data (from extract_and_cache) to avoid re-extracting features.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"FEDERATED TRAINING — {partition.upper()} partition")
    print(f"{'='*60}")
    print(f"Clients: {num_clients} | Rounds: {num_rounds} | "
          f"Local epochs: {local_epochs} | LR: {lr}")
    print(f"Device: {device}")

    # ── Load/cache features ────────────────────────────────────────
    if cached_data is None:
        cached_data = extract_and_cache(data_dir, device, num_workers)

    train_dataset = cached_data["train_dataset"]
    train_features = cached_data["train_features"]
    train_labels = cached_data["train_labels"]
    class_names = cached_data["class_names"]

    val_feature_loader = DataLoader(
        TensorDataset(cached_data["val_features"], cached_data["val_labels"]),
        batch_size=512, shuffle=False, pin_memory=True,
    )

    # ── Partition data ─────────────────────────────────────────────
    if partition == "iid":
        client_indices = partition_iid(train_dataset, num_clients, seed=seed)
    elif partition == "non_iid":
        client_indices = partition_non_iid(
            train_dataset, num_clients, classes_per_client=classes_per_client, seed=seed
        )
    elif partition == "dirichlet":
        client_indices = partition_non_iid_dirichlet(
            train_dataset, num_clients, alpha=alpha, seed=seed
        )
    else:
        raise ValueError(f"Unknown partition strategy: {partition}")

    print(f"\nData partition ({partition}):")
    print_partition_stats(train_dataset, client_indices, class_names)

    # ── Create clients and server ──────────────────────────────────
    client_loaders = create_client_feature_dataloaders(
        train_features, train_labels, client_indices, batch_size=batch_size
    )

    clients = {}
    for cid, loader in client_loaders.items():
        clients[cid] = FederatedClient(cid, loader, device, num_classes=num_classes)

    server = FederatedServer(num_classes=num_classes, device=device)

    # ── Training loop ──────────────────────────────────────────────
    history = {
        "global_val_acc": [],
        "global_val_loss": [],
        "client_metrics": {cid: [] for cid in clients},
    }

    for rnd in range(1, num_rounds + 1):
        round_start = time.time()

        global_fc_params = server.get_global_fc_params()

        client_updates = []
        for cid, client in clients.items():
            result = client.local_train(global_fc_params, local_epochs=local_epochs, lr=lr)
            client_updates.append(result)
            history["client_metrics"][cid].append(result["metrics"][-1])

        server.aggregate(client_updates)

        val_metrics = server.evaluate(val_feature_loader)
        history["global_val_acc"].append(val_metrics["accuracy"])
        history["global_val_loss"].append(val_metrics["loss"])

        round_time = time.time() - round_start
        client_accs = [u["metrics"][-1]["accuracy"] for u in client_updates]

        print(
            f"Round {rnd:3d}/{num_rounds} | "
            f"Val Acc: {val_metrics['accuracy']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Client Accs: [{', '.join(f'{a:.1f}' for a in client_accs)}] | "
            f"Time: {round_time:.1f}s"
        )

    # ── Store config ──────────────────────────────────────────────
    history["client_indices"] = client_indices
    history["config"] = {
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "partition": partition,
        "alpha": alpha,
        "lr": lr,
        "batch_size": batch_size,
    }

    print(f"\nFinal global validation accuracy: {history['global_val_acc'][-1]:.2f}%")
    return history, server


# ── Centralized Baseline ─────────────────────────────────────────────────

def run_centralized_baseline(
    num_epochs=20,
    batch_size=512,
    lr=1e-3,
    num_workers=4,
    data_dir="satellite_data",
    num_classes=10,
    device=None,
    cached_data=None,
):
    """Centralized training using cached features for fair comparison."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print("CENTRALIZED BASELINE TRAINING")
    print(f"{'='*60}")
    print(f"Epochs: {num_epochs} | Batch size: {batch_size} | LR: {lr}")
    print(f"Device: {device}")

    if cached_data is None:
        cached_data = extract_and_cache(data_dir, device, num_workers)

    train_loader = DataLoader(
        TensorDataset(cached_data["train_features"], cached_data["train_labels"]),
        batch_size=batch_size, shuffle=True, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(cached_data["val_features"], cached_data["val_labels"]),
        batch_size=batch_size, shuffle=False, pin_memory=True,
    )

    model = FCClassifier(in_features=2048, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}

    for epoch in range(1, num_epochs + 1):
        train_m = train_fc_epoch(model, train_loader, optimizer, criterion, device)
        val_m = validate_fc(model, val_loader, criterion, device)

        history["train_acc"].append(train_m["accuracy"])
        history["train_loss"].append(train_m["loss"])
        history["val_acc"].append(val_m["accuracy"])
        history["val_loss"].append(val_m["loss"])

        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train Acc: {train_m['accuracy']:.2f}% | "
            f"Val Acc: {val_m['accuracy']:.2f}% | "
            f"Val Loss: {val_m['loss']:.4f}"
        )

    print(f"\nBest centralized val accuracy: {max(history['val_acc']):.2f}%")
    return history, model


# ── Plotting Functions ───────────────────────────────────────────────────

def plot_global_accuracy(history, title="Global Validation Accuracy per Round"):
    """Plot global validation accuracy over federated rounds."""
    rounds = range(1, len(history["global_val_acc"]) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, history["global_val_acc"], "b-o", markersize=4, label="Global Val Acc")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_client_curves(history, title="Per-Client Training Accuracy"):
    """Plot per-client training accuracy across rounds."""
    plt.figure(figsize=(10, 5))
    for cid, metrics_list in history["client_metrics"].items():
        accs = [m["accuracy"] for m in metrics_list]
        rounds = range(1, len(accs) + 1)
        plt.plot(rounds, accs, "-o", markersize=3, label=f"Client {cid}")
    plt.xlabel("Communication Round")
    plt.ylabel("Local Training Accuracy (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_client_class_distribution(dataset, client_indices, class_names=None):
    """Bar chart of per-client class distributions."""
    targets = np.array([y for _, y in dataset]) if not hasattr(dataset, "targets") \
        else np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    if class_names is None:
        class_names = [str(c) for c in range(num_classes)]

    num_clients = len(client_indices)
    fig, axes = plt.subplots(1, num_clients, figsize=(4 * num_clients, 4), sharey=True)
    if num_clients == 1:
        axes = [axes]

    for ax, (cid, indices) in zip(axes, sorted(client_indices.items())):
        labels = targets[indices]
        counts = np.bincount(labels, minlength=num_classes)
        ax.bar(range(num_classes), counts, color="steelblue")
        ax.set_title(f"Client {cid}\n({len(indices)} samples)")
        ax.set_xlabel("Class")
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("Sample Count")
    plt.suptitle("Client Data Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_comparison(centralized_history, federated_histories, labels=None):
    """Compare centralized vs federated validation accuracy curves."""
    if labels is None:
        labels = [f"Federated {i}" for i in range(len(federated_histories))]

    plt.figure(figsize=(10, 6))

    epochs = range(1, len(centralized_history["val_acc"]) + 1)
    plt.plot(epochs, centralized_history["val_acc"], "k-s", markersize=4,
             label="Centralized", linewidth=2)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for i, (hist, label) in enumerate(zip(federated_histories, labels)):
        rounds = range(1, len(hist["global_val_acc"]) + 1)
        plt.plot(rounds, hist["global_val_acc"], f"-o", color=colors[i % len(colors)],
                 markersize=4, label=label)

    plt.xlabel("Epoch / Communication Round")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Centralized vs Federated Training Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_comparison_table(centralized_history, federated_histories, labels=None):
    """Print a summary comparison table."""
    if labels is None:
        labels = [f"Federated {i}" for i in range(len(federated_histories))]

    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'Best Val Acc':<15} {'Final Val Acc':<15} {'Rounds/Epochs':<15}")
    print(f"{'-'*70}")

    best_c = max(centralized_history["val_acc"])
    final_c = centralized_history["val_acc"][-1]
    n_c = len(centralized_history["val_acc"])
    print(f"{'Centralized':<30} {best_c:<15.2f} {final_c:<15.2f} {n_c:<15}")

    for hist, label in zip(federated_histories, labels):
        best_f = max(hist["global_val_acc"])
        final_f = hist["global_val_acc"][-1]
        n_f = len(hist["global_val_acc"])
        detail = f"{n_f} rounds"
        print(f"{label:<30} {best_f:<15.2f} {final_f:<15.2f} {detail:<15}")

    print(f"{'='*70}")


def print_communication_cost(history):
    """Estimate and print communication cost for a federated run."""
    cfg = history.get("config", {})
    num_clients = cfg.get("num_clients", 5)
    num_rounds = cfg.get("num_rounds", 20)

    fc_params = 2048 * 10 + 10  # 20,490
    bytes_per_param = 4  # float32
    fc_size_bytes = fc_params * bytes_per_param
    fc_size_kb = fc_size_bytes / 1024

    per_round_kb = num_clients * fc_size_kb * 2
    total_kb = per_round_kb * num_rounds
    total_mb = total_kb / 1024

    print(f"\n{'='*50}")
    print("COMMUNICATION COST ANALYSIS")
    print(f"{'='*50}")
    print(f"FC layer size: {fc_params:,} params ({fc_size_kb:.1f} KB)")
    print(f"Per round: {num_clients} clients x {fc_size_kb:.1f} KB x 2 = {per_round_kb:.1f} KB")
    print(f"Total: {num_rounds} rounds x {per_round_kb:.1f} KB = {total_mb:.2f} MB")
    print(f"{'='*50}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract features once, reuse for all experiments
    cached_data = extract_and_cache("satellite_data", device)

    # Quick test: 3 clients, 3 rounds, IID
    history, server = run_federated(
        num_clients=3, num_rounds=3, local_epochs=2,
        partition="iid", batch_size=512, lr=1e-3,
        cached_data=cached_data,
    )
    plot_global_accuracy(history)
