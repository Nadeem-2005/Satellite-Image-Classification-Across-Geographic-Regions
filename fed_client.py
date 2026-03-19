"""
Federated Client (Optimized with Feature Caching)

Each client holds a lightweight FC classifier (~20K params) instead of
a full ResNet-50. Trains on cached 2048-dim backbone features, eliminating
redundant CNN forward passes and yielding ~50-100x speedup.
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from model import FCClassifier, train_fc_epoch


class FederatedClient:
    """A single federated learning client using cached features."""

    def __init__(self, client_id, dataloader, device, num_classes=10):
        """
        Args:
            client_id: Unique identifier for this client.
            dataloader: DataLoader of cached (features, labels) tensors.
            device: torch.device to use (cpu or cuda).
            num_classes: Number of output classes.
        """
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.num_classes = num_classes

        # Lightweight FC-only model (no ResNet-50 backbone needed)
        self.model = FCClassifier(in_features=2048, num_classes=num_classes).to(device)
        self.num_samples = len(dataloader.dataset)

    def get_fc_params(self):
        """Extract the FC layer parameters as an OrderedDict."""
        return OrderedDict({
            "fc.weight": self.model.fc.weight.data.clone().cpu(),
            "fc.bias": self.model.fc.bias.data.clone().cpu(),
        })

    def set_fc_params(self, fc_state):
        """Load global FC parameters into the local model."""
        self.model.fc.weight.data.copy_(fc_state["fc.weight"].to(self.device))
        self.model.fc.bias.data.copy_(fc_state["fc.bias"].to(self.device))

    def local_train(self, global_fc_params, local_epochs=5, lr=1e-3):
        """
        Perform local training starting from the global FC parameters.

        Returns:
            dict with 'fc_params', 'num_samples', and 'metrics'.
        """
        self.set_fc_params(global_fc_params)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        epoch_metrics = []
        for _ in range(local_epochs):
            metrics = train_fc_epoch(
                self.model, self.dataloader, optimizer, criterion, self.device
            )
            epoch_metrics.append(metrics)

        return {
            "fc_params": self.get_fc_params(),
            "num_samples": self.num_samples,
            "metrics": epoch_metrics,
        }
