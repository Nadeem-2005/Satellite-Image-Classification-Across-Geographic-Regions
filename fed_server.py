"""
Federated Server (Optimized with Feature Caching)

Maintains a lightweight FC global model, broadcasts FC parameters to clients,
aggregates client updates using FedAvg, and evaluates on cached features.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models

from model import FCClassifier, validate_fc


class FederatedServer:
    """Central server for Federated Averaging."""

    def __init__(self, num_classes=10, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.num_classes = num_classes

        # Lightweight FC-only global model
        self.model = FCClassifier(in_features=2048, num_classes=num_classes).to(device)

    def get_global_fc_params(self):
        """Get the current global FC layer parameters for broadcasting."""
        return OrderedDict({
            "fc.weight": self.model.fc.weight.data.clone().cpu(),
            "fc.bias": self.model.fc.bias.data.clone().cpu(),
        })

    def aggregate(self, client_updates):
        """
        FedAvg aggregation — weighted average of client FC parameters
        proportional to each client's sample count.
        """
        total_samples = sum(u["num_samples"] for u in client_updates)

        agg_params = OrderedDict()
        for key in client_updates[0]["fc_params"]:
            agg_params[key] = torch.zeros_like(client_updates[0]["fc_params"][key])

        for update in client_updates:
            weight = update["num_samples"] / total_samples
            for key in agg_params:
                agg_params[key] += weight * update["fc_params"][key]

        self.model.fc.weight.data.copy_(agg_params["fc.weight"].to(self.device))
        self.model.fc.bias.data.copy_(agg_params["fc.bias"].to(self.device))

        return agg_params

    def evaluate(self, val_loader):
        """Evaluate the global model on cached validation features."""
        criterion = nn.CrossEntropyLoss()
        return validate_fc(self.model, val_loader, criterion, self.device)

    def save_model(self, path, class_names=None, history=None):
        """
        Save a full ResNet-50 checkpoint with the trained FC head.
        Reconstructs the backbone for inference compatibility.
        """
        full_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in full_model.parameters():
            param.requires_grad = False
        full_model.fc = nn.Linear(2048, self.num_classes)
        full_model.fc.weight.data.copy_(self.model.fc.weight.data.cpu())
        full_model.fc.bias.data.copy_(self.model.fc.bias.data.cpu())

        checkpoint = {
            "model_state_dict": full_model.state_dict(),
            "fc_state_dict": OrderedDict({
                "fc.weight": self.model.fc.weight.data.cpu(),
                "fc.bias": self.model.fc.bias.data.cpu(),
            }),
            "num_classes": self.num_classes,
        }
        if class_names is not None:
            checkpoint["class_names"] = class_names
        if history is not None:
            checkpoint["history"] = history
        torch.save(checkpoint, path)
        print(f"Global model saved to {path}")
