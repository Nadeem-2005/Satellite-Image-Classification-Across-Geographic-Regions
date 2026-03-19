"""
Federated Data Partitioning

Splits a training dataset across N simulated clients to emulate
geographically distributed satellite imagery sources. Supports:
  - IID partitioning (uniform random split)
  - Non-IID shard-based partitioning (each client gets a few classes)
  - Non-IID Dirichlet partitioning (realistic label heterogeneity)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from collections import defaultdict


def _get_targets(dataset):
    """Extract integer labels from a dataset."""
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    return np.array([y for _, y in dataset])


# ── IID Partitioning ──────────────────────────────────────────────────────

def partition_iid(dataset, num_clients, seed=42):
    """
    Random equal split — each client gets ~uniform class distribution.

    Args:
        dataset: PyTorch dataset with labels.
        num_clients: Number of simulated clients.
        seed: Random seed for reproducibility.

    Returns:
        dict mapping client_id (int) → list of sample indices.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))
    shards = np.array_split(indices, num_clients)
    return {i: shard.tolist() for i, shard in enumerate(shards)}


# ── Non-IID Shard-Based Partitioning ─────────────────────────────────────

def partition_non_iid(dataset, num_clients, classes_per_client=2, seed=42):
    """
    Shard-based non-IID split — each client receives only a few classes.

    Args:
        dataset: PyTorch dataset with labels.
        num_clients: Number of simulated clients.
        classes_per_client: How many classes each client receives.
        seed: Random seed for reproducibility.

    Returns:
        dict mapping client_id (int) → list of sample indices.
    """
    rng = np.random.default_rng(seed)
    targets = _get_targets(dataset)
    num_classes = len(np.unique(targets))

    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    # Shuffle within each class
    for c in class_indices:
        rng.shuffle(class_indices[c])

    # Total shards = num_clients * classes_per_client
    total_shards = num_clients * classes_per_client

    # Split each class into (total_shards / num_classes) equal pieces
    shards_per_class = total_shards // num_classes
    all_shards = []
    for c in range(num_classes):
        splits = np.array_split(class_indices[c], shards_per_class)
        for s in splits:
            all_shards.append(s.tolist())

    # Shuffle shards and assign classes_per_client shards to each client
    rng.shuffle(all_shards)
    client_indices = {}
    for i in range(num_clients):
        client_indices[i] = []
        for j in range(classes_per_client):
            shard_idx = i * classes_per_client + j
            if shard_idx < len(all_shards):
                client_indices[i].extend(all_shards[shard_idx])
    return client_indices


# ── Non-IID Dirichlet Partitioning ───────────────────────────────────────

def partition_non_iid_dirichlet(dataset, num_clients, alpha=0.5, seed=42):
    """
    Dirichlet distribution-based non-IID split for realistic heterogeneity.

    Lower alpha → more heterogeneous (alpha=0.1 extreme, alpha=0.5 moderate).

    Args:
        dataset: PyTorch dataset with labels.
        num_clients: Number of simulated clients.
        alpha: Dirichlet concentration parameter.
        seed: Random seed for reproducibility.

    Returns:
        dict mapping client_id (int) → list of sample indices.
    """
    rng = np.random.default_rng(seed)
    targets = _get_targets(dataset)
    num_classes = len(np.unique(targets))

    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    client_indices = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        indices_c = np.array(class_indices[c])
        rng.shuffle(indices_c)

        # Draw a Dirichlet distribution over clients for this class
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # Convert proportions to counts (ensuring all samples are assigned)
        counts = (proportions * len(indices_c)).astype(int)
        # Assign any remainder to the client with the largest proportion
        remainder = len(indices_c) - counts.sum()
        counts[np.argmax(proportions)] += remainder

        start = 0
        for i in range(num_clients):
            client_indices[i].extend(indices_c[start : start + counts[i]].tolist())
            start += counts[i]

    return client_indices


# ── Client DataLoaders ───────────────────────────────────────────────────

def create_client_dataloaders(dataset, client_indices, batch_size=32, num_workers=2):
    """
    Wrap each client's subset of the dataset into a DataLoader.

    Args:
        dataset: Full training dataset.
        client_indices: dict mapping client_id → list of sample indices.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of data loading workers.

    Returns:
        dict mapping client_id → DataLoader.
    """
    client_loaders = {}
    for client_id, indices in client_indices.items():
        subset = Subset(dataset, indices)
        client_loaders[client_id] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    return client_loaders


# ── Partition Statistics ─────────────────────────────────────────────────

def print_partition_stats(dataset, client_indices, class_names=None):
    """
    Print per-client class distribution table.

    Args:
        dataset: Full training dataset.
        client_indices: dict mapping client_id → list of sample indices.
        class_names: Optional list of class name strings.
    """
    targets = _get_targets(dataset)
    num_classes = len(np.unique(targets))
    if class_names is None:
        class_names = [str(c) for c in range(num_classes)]

    num_clients = len(client_indices)

    # Header
    header = f"{'Client':<10}" + "".join(f"{name:<16}" for name in class_names) + f"{'Total':<10}"
    print("=" * len(header))
    print("Per-Client Class Distribution")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for client_id in sorted(client_indices.keys()):
        indices = client_indices[client_id]
        client_labels = targets[indices]
        counts = np.bincount(client_labels, minlength=num_classes)
        row = f"{client_id:<10}" + "".join(f"{c:<16}" for c in counts) + f"{len(indices):<10}"
        print(row)

    # Totals
    print("-" * len(header))
    all_indices = [idx for idxs in client_indices.values() for idx in idxs]
    all_labels = targets[all_indices]
    totals = np.bincount(all_labels, minlength=num_classes)
    row = f"{'Total':<10}" + "".join(f"{c:<16}" for c in totals) + f"{len(all_indices):<10}"
    print(row)
    print("=" * len(header))


# ── Feature-Based Client DataLoaders ─────────────────────────────────

def create_client_feature_dataloaders(features, labels, client_indices, batch_size=512):
    """
    Create DataLoaders from cached feature vectors for each client.
    No num_workers needed since data is already in memory.
    """
    client_loaders = {}
    for cid, indices in client_indices.items():
        idx = torch.tensor(indices, dtype=torch.long)
        ds = TensorDataset(features[idx], labels[idx])
        client_loaders[cid] = DataLoader(
            ds, batch_size=batch_size, shuffle=True, pin_memory=True,
        )
    return client_loaders
