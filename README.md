# Cross-Domain Federated Transfer Learning for Satellite Image Classification Across Geographic Regions

A privacy-preserving federated learning system for satellite image classification using the EuroSAT dataset. The project implements **Federated Averaging (FedAvg)** with a pretrained **ResNet-50** backbone, enabling multiple simulated geographic regions (clients) to collaboratively train a global classifier without sharing raw satellite imagery.

Only the fully-connected classification head (~80 KB) is communicated each round, keeping the frozen convolutional backbone entirely local.

---

## Architecture

```
                        ┌─────────────────────┐
                        │   Federated Server   │
                        │  (Global FC Params)  │
                        └──────────┬──────────┘
                  ┌────────────────┼────────────────┐
           broadcast FC      broadcast FC      broadcast FC
                  │                │                │
            ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
            │  Client 1  │   │  Client 2  │   │  Client N  │
            │ (Region A) │   │ (Region B) │   │ (Region N) │
            │            │   │            │   │            │
            │ ResNet-50  │   │ ResNet-50  │   │ ResNet-50  │
            │ (frozen)   │   │ (frozen)   │   │ (frozen)   │
            │ + FC head  │   │ + FC head  │   │ + FC head  │
            └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
                  │                │                │
           upload FC params  upload FC params  upload FC params
                  │                │                │
                  └────────────────┼────────────────┘
                                   ▼
                        ┌─────────────────────┐
                        │   FedAvg Aggregate   │
                        │ (weighted by sample  │
                        │       count)         │
                        └─────────────────────┘
```

**Per round:** Server broadcasts global FC parameters → each client trains locally for E epochs → clients upload updated FC parameters → server aggregates via weighted averaging → repeat.

---

## Dataset

**EuroSAT RGB** — 27,000 geo-referenced Sentinel-2 satellite image patches (64x64 RGB), resized to 224x224 for ResNet-50 input.

| Class | Training Samples | Validation Samples |
|---|---|---|
| AnnualCrop | 2,400 | 600 |
| Forest | 2,400 | 600 |
| HerbaceousVegetation | 2,400 | 600 |
| Highway | 2,000 | 500 |
| Industrial | 2,000 | 500 |
| Pasture | 1,600 | 400 |
| PermanentCrop | 2,000 | 500 |
| Residential | 2,400 | 600 |
| River | 2,000 | 500 |
| SeaLake | 2,400 | 600 |
| **Total** | **20,200** | **6,800** |

Source: [Zenodo — EuroSAT RGB](https://zenodo.org/records/7711810)

---

## Project Structure

```
├── Datasets/
│   └── EuroSAT_RGB/              # Raw dataset (10 class folders, 27K images)
├── satellite_data/
│   ├── train/                    # 80% split (20,200 images)
│   └── val/                      # 20% split (6,800 images)
├── Colab Results/
│   └── resnet50_eurosat.pth      # Saved model checkpoint (~90 MB)
│
├── prepare_data.py               # Dataset splitting script
├── data_loader.py                # Transforms, dataset loading, DataLoaders
├── model.py                      # ResNet-50 model definition & train/val loops
├── fed_data.py                   # Federated data partitioning strategies
├── fed_client.py                 # Federated client implementation
├── fed_server.py                 # Federated server with FedAvg aggregation
├── fed_train.py                  # Main training orchestration & visualization
│
├── Federated_Training.ipynb                        # Full federated experiments (Colab)
├── Federated_Transfer_Learning_Satellite.ipynb     # Centralized baseline (Colab)
│
└── requirements.txt              # Python dependencies
```

---

## File Descriptions

### `prepare_data.py`
Splits the raw EuroSAT_RGB dataset into an 80/20 train/validation directory structure. Reads images from `Datasets/EuroSAT_RGB/{class}/` and copies them into `satellite_data/train/{class}/` and `satellite_data/val/{class}/` using a fixed random seed (42) for reproducibility.

### `data_loader.py`
Handles image preprocessing and PyTorch DataLoader creation:
- Resizes images to 224x224
- Normalizes with ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- Wraps datasets with `torchvision.datasets.ImageFolder`
- Creates DataLoaders with configurable batch size and worker count

### `model.py`
Defines the ResNet-50 transfer learning setup:
- Loads ImageNet-pretrained ResNet-50
- Freezes all backbone layers (~23.5M parameters)
- Replaces the final FC layer with `nn.Linear(2048, 10)` (~20,490 trainable parameters)
- Provides `train_one_epoch()` and `validate()` functions used by both centralized and federated training

### `fed_data.py`
Implements three data partitioning strategies to simulate heterogeneous geographic regions:

| Strategy | Function | Description |
|---|---|---|
| **IID** | `partition_iid()` | Uniform random split — each client gets a balanced class distribution |
| **Non-IID (Shard)** | `partition_non_iid()` | Each client receives only a fixed number of classes (default: 2) |
| **Non-IID (Dirichlet)** | `partition_non_iid_dirichlet()` | Dirichlet-sampled label heterogeneity — lower alpha = more skewed |

Also provides `create_client_dataloaders()` to wrap each client's data subset into a DataLoader, and `print_partition_stats()` for debugging class distributions.

### `fed_client.py`
Implements the `FederatedClient` class representing a single geographic region:
- Maintains a local copy of ResNet-50 (frozen backbone + trainable FC head)
- `local_train()` — receives global FC parameters, trains locally for E epochs using Adam optimizer, returns updated FC parameters and sample count
- `get_fc_params()` / `set_fc_params()` — extracts/loads only the FC layer state dict for communication

### `fed_server.py`
Implements the `FederatedServer` class as the central coordinator:
- `get_global_fc_params()` — broadcasts current global FC parameters to all clients
- `aggregate()` — performs FedAvg: weighted average of client FC parameters by sample count
- `evaluate()` — validates the global model on the full validation set
- `save_model()` — saves a checkpoint with model weights, class names, and training history

### `fed_train.py`
Main orchestration script containing:
- **`run_federated()`** — full FedAvg training loop across configurable rounds and clients
- **`run_centralized_baseline()`** — standard centralized training for comparison
- **Visualization functions:** global accuracy curves, per-client training curves, client class distribution charts, centralized vs. federated comparison plots
- **`print_communication_cost()`** — estimates total communication overhead

### Jupyter Notebooks

- **`Federated_Training.ipynb`** — Complete self-contained Colab notebook that downloads EuroSAT, runs centralized baseline (20 epochs), federated IID (5 clients, 20 rounds), and federated non-IID (Dirichlet α=0.5), then compares results with plots and tables.
- **`Federated_Transfer_Learning_Satellite.ipynb`** — Simpler notebook focused on centralized transfer learning baseline only (10 epochs, achieves ~94.5% validation accuracy).

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU recommended (CPU works but is significantly slower)

### Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` contains:
```
torch
torchvision
```

You will also need `numpy` and `matplotlib` (installed automatically as dependencies of torch/torchvision).

### Download the Dataset

**Option A — Manual download:**

1. Download the EuroSAT RGB dataset from [Zenodo](https://zenodo.org/records/7711810)
2. Extract into `Datasets/EuroSAT_RGB/` so the structure is:
   ```
   Datasets/EuroSAT_RGB/
   ├── AnnualCrop/
   ├── Forest/
   ├── HerbaceousVegetation/
   ...
   └── SeaLake/
   ```

**Option B — Use the Colab notebook:**

The `Federated_Training.ipynb` notebook automatically downloads and extracts the dataset.

### Prepare Train/Val Split

```bash
python prepare_data.py
```

This creates the `satellite_data/train/` and `satellite_data/val/` directories with an 80/20 split.

---

## Run Commands

### Run Federated Training (Local)

```bash
python fed_train.py
```

By default this runs a quick experiment with 3 clients, 3 rounds, and 2 local epochs. To customize, edit the `__main__` block in `fed_train.py` or import and call `run_federated()` directly:

```python
from fed_train import run_federated, plot_global_accuracy

# Full IID experiment
history, server = run_federated(
    num_clients=5,
    num_rounds=20,
    local_epochs=5,
    partition="iid",
    batch_size=32,
    lr=1e-3,
    data_dir="satellite_data",
)
plot_global_accuracy(history)

# Non-IID with Dirichlet partitioning
history_noniid, server = run_federated(
    num_clients=5,
    num_rounds=20,
    local_epochs=5,
    partition="dirichlet",
    alpha=0.5,
    batch_size=32,
    lr=1e-3,
)

# Shard-based non-IID (each client gets 2 classes)
history_shard, server = run_federated(
    num_clients=5,
    num_rounds=20,
    local_epochs=5,
    partition="non_iid",
    classes_per_client=2,
)
```

### Run Centralized Baseline

```python
from fed_train import run_centralized_baseline

history = run_centralized_baseline(
    num_epochs=20,
    batch_size=32,
    lr=1e-3,
    data_dir="satellite_data",
)
```

### Compare Federated vs. Centralized

```python
from fed_train import plot_comparison, print_comparison_table

plot_comparison(
    centralized_history,
    [history_iid, history_noniid],
    labels=["Federated IID", "Federated Non-IID"],
)
print_comparison_table(centralized_history, [history_iid, history_noniid])
```

### Run on Google Colab

1. Upload `Federated_Training.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells sequentially — the notebook handles dataset download, splitting, training, and visualization

---

## Configuration & Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `num_clients` | 5 | Number of simulated federated clients (geographic regions) |
| `num_rounds` | 20 | Number of communication rounds |
| `local_epochs` | 5 | Training epochs per client per round |
| `partition` | `"iid"` | Data partition strategy: `"iid"`, `"non_iid"`, or `"dirichlet"` |
| `alpha` | 0.5 | Dirichlet concentration parameter (lower = more heterogeneous) |
| `classes_per_client` | 2 | Number of classes per client (shard-based non-IID only) |
| `batch_size` | 32 | Mini-batch size for training |
| `lr` | 1e-3 | Learning rate (Adam optimizer) |
| `num_workers` | 2 | DataLoader worker processes |
| `num_classes` | 10 | Number of classification classes |
| `seed` | 42 | Random seed for reproducibility |

### Model Details

| Property | Value |
|---|---|
| Backbone | ResNet-50 (ImageNet pretrained) |
| Frozen parameters | ~23.5M |
| Trainable parameters (FC layer) | 20,490 |
| Input size | 224 x 224 x 3 (RGB) |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |

---

## Communication Cost

Since only the FC layer is communicated, the bandwidth requirement is minimal:

```
FC layer size: (2048 × 10) weights + 10 biases = 20,490 float32 parameters
Per transfer:  20,490 × 4 bytes ≈ 80 KB
Per round:     num_clients × 80 KB × 2 (download + upload)
Example:       5 clients × 80 KB × 2 = 800 KB/round
Full training: 20 rounds × 800 KB = 16 MB total
```

This is orders of magnitude smaller than transmitting the full ResNet-50 model (~90 MB) or raw images.

---

## Expected Results

| Training Mode | Validation Accuracy |
|---|---|
| Centralized (20 epochs) | ~94.5% |
| Federated IID (5 clients, 20 rounds) | ~92-94% |
| Federated Non-IID Dirichlet α=0.5 (5 clients, 20 rounds) | ~88-92% |

The federated IID setting closely approaches centralized performance, while non-IID partitioning shows a modest accuracy drop due to data heterogeneity across clients.

---

## Key Concepts

- **Federated Learning:** Clients train locally on private data and only share model parameter updates — raw data never leaves the client.
- **Transfer Learning:** A pretrained ImageNet backbone extracts general visual features; only the task-specific classification head is fine-tuned.
- **FedAvg:** The server computes a weighted average of client parameters, where each client's contribution is proportional to its dataset size.
- **Non-IID Data:** Real-world geographic regions have different land-use distributions. Dirichlet-based partitioning simulates this heterogeneity realistically.
