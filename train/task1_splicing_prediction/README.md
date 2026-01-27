# Task 1: Splicing Prediction Training

Multi-class classification model to predict splicing sites using pre-extracted DNA sequence embeddings from Nucleotide Transformer (NT).

## 📋 Overview

This pipeline performs:
1. **Data Preparation**: Organize embeddings from pre-computed DNA sequence representations
2. **Model Training**: Train a classifier on splicing site prediction with multiple dataset ratios
3. **Experiment Tracking**: Automatically save config, results, and model checkpoints for each experiment
4. **Performance Monitoring**: Real-time TensorBoard logging with comprehensive metrics

## 🏗️ Model Architecture

```
Input: DNA Sequence Embeddings [batch_size, embedding_dim]
    ↓
Projection Layer: Linear (embedding_dim → hidden_dims[0])
    ↓
Hidden Layers: [512, 256] with ReLU, Batch Norm, Dropout
    ↓
Output Layer: Linear → 3 classes (Splicing Site Classes)
    ↓
Classification Output: Probability scores per class
```

### Key Features:
- **Batch Normalization**: Stabilizes training
- **Dropout**: Regularization (default: 0.3)
- **Gradient Clipping**: Prevents exploding gradients (max_norm=1.0)
- **L2 Regularization (Weight Decay)**: Prevent overfitting (default: 1e-4)
- **Confusion Matrix Visualization**: In TensorBoard per epoch

## 📊 Comprehensive Metrics

The training pipeline tracks:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **Specificity**: True negatives / (true negatives + false positives)
- **F1 Score**: Harmonic mean of precision and recall
- **MCC**: Matthews Correlation Coefficient (robust multi-class metric)
- **AUC**: Area under ROC curve (one-vs-rest)
- **Balanced Accuracy**: Macro average of recall per class

## 📁 Directory Structure

```
task1_splicing_prediction/
├── README.md                    # This file
├── data_preparation/            # Data preprocessing
│   ├── data_prepare.ipynb       # Data preparation notebook
│   ├── train_test_split.py      # Data splitting logic
│   └── train_val/               # Split training/validation data
│       ├── ratio_10_80_10/      # Train:Val:Test = 10:80:10
│       │   ├── set_1/
│       │   │   ├── train_embeddings.pt
│       │   │   └── val_embeddings.pt
│       │   └── set_2/...
│       ├── ratio_20_60_20/...
│       └── ...
│
└── training/                    # Training & evaluation
    ├── config.py                # Centralized hyperparameters
    ├── dataset.py               # PyTorch Dataset class
    ├── model.py                 # SpliceSiteClassifier model
    ├── metrics.py               # Metric calculations (torchmetrics)
    ├── train_set.py             # Training loop with experiment tracking
    ├── train.py                 # Main entry point with CLI
    ├── cm_visualize.py          # Confusion matrix visualization
    ├── fileio.py                # File I/O utilities
    ├── main.ipynb               # Notebook to run full pipeline
    │
    └── trained_models/
        └── experiments/         # All experiments (auto-created)
            ├── experiment_1/
            │   ├── config.json              # Hyperparameters snapshot
            │   ├── config.py                # Copy of config.py (reproducibility)
            │   ├── args.json                # Command-line arguments
            │   ├── ratio_10_80_10/
            │   │   ├── set_1/
            │   │   │   ├── best_model.pt    # Best model checkpoint
            │   │   │   ├── confusion_matrix.png
            │   │   │   ├── model_summary.txt
            │   │   │   ├── results.json     # Metrics & training history
            │   │   │   └── tensorboard/     # TensorBoard logs
            │   │   └── set_2/...
            │   └── ratio_20_60_20/...
            └── experiment_2/...
```

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch (with CUDA if GPU available)
- Dependencies: `pandas`, `numpy`, `torchmetrics`, `tensorboard`, `tqdm`, `scikit-learn`, `matplotlib`, `seaborn`, `torchinfo`

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install torchmetrics tensorboard tqdm scikit-learn matplotlib seaborn torchinfo
```

## 📖 Usage Guide

### Method 1: Using Jupyter Notebook (Recommended)

Navigate to `training/` directory and open `main.ipynb`:

```python
# Cell 1: Configuration & Setup
# Loads config and shows device info

# Cell 2: Train Models
# Runs training with experiment tracking

# Cell 3: View Results
# Displays experiment results summary

# Cell 4: Visualization
# Plot metrics across experiments
```

### Method 2: Using Command Line

```bash
cd training/

# Train with default settings
python train.py

# Train with custom settings
python train.py --batch-size 32 --lr 5e-4 --epochs 100 --patience 10

# Train specific data ratio only
python train.py --ratio ratio_10_80_10 --set set_1

# Specify GPU
python train.py --device cuda --seed 42
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch-size` | int | 64 | Batch size for training |
| `--lr` | float | 1e-3 | Learning rate |
| `--weight-decay` | float | 1e-4 | L2 regularization coefficient |
| `--epochs` | int | 50 | Maximum number of epochs |
| `--patience` | int | 7 | Early stopping patience |
| `--dropout` | float | 0.3 | Dropout rate |
| `--device` | str | cuda/cpu | Device to use (auto-detected) |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--data-dir` | str | auto | Path to training data |
| `--save-root` | str | auto | Root directory for experiments |
| `--exp-num` | int | auto | Experiment number (auto-incremented) |
| `--ratio` | str | None | Train specific ratio (e.g., `ratio_10_80_10`) |
| `--set` | str | None | Train specific set (e.g., `set_1`) |

## 🎯 Training Configuration

Default hyperparameters (from `config.py`):

```python
BATCH_SIZE = 64              # Training batch size
LR = 1e-3                    # Learning rate (Adam optimizer)
WEIGHT_DECAY = 1e-4          # L2 regularization
EPOCHS = 50                  # Maximum training epochs
PATIENCE = 7                 # Early stopping patience
DROPOUT = 0.3                # Dropout rate for regularization
HIDDEN_DIMS = [512, 256]     # MLP hidden layer dimensions
NUM_CLASSES = 3              # Number of splicing site classes
SEED = 42                    # Random seed
```

## 📊 Experiment Tracking

Each experiment automatically creates a structured directory with:

```
experiment_N/
├── config.json              # JSON snapshot of all hyperparameters
├── config.py                # Copy of config.py (for reproducibility)
├── args.json                # Command-line arguments used
└── ratio_X_Y_Z/
    └── set_M/
        ├── best_model.pt                    # Best model weights
        ├── confusion_matrix.png             # Final confusion matrix
        ├── model_summary.txt                # Model architecture info
        ├── results.json                     # Best metrics & training history
        └── tensorboard/
            └── events.out.tfevents.*        # TensorBoard event files
```

### Accessing Results

```python
import json

# Load best metrics
with open("trained_models/experiments/experiment_1/ratio_10_80_10/set_1/results.json") as f:
    results = json.load(f)
    print(results['best_metrics'])
    # {'accuracy': 0.92, 'precision': 0.89, 'recall': 0.88, ...}

# Load training history
training_history = results['training_history']
print(f"Epochs trained: {len(training_history['train_loss'])}")
```

## 📈 Model Summary & Visualization

### Model Architecture Summary

Each training run automatically saves `model_summary.txt` with:
- Input/output shapes
- Number of parameters per layer
- Total model size

Example:
```
Layer         Input Size     Output Size    Params
Linear        [batch, 1024]  [batch, 512]   525,312
BatchNorm1d   [batch, 512]   [batch, 512]   1,024
ReLU          [batch, 512]   [batch, 512]   0
Dropout       [batch, 512]   [batch, 512]   0
...
Total params: 2,043,655
```

### Confusion Matrix Visualization

Confusion matrices are:
1. **Saved as PNG**: `confusion_matrix.png` 
2. **Logged to TensorBoard**: View per epoch in browser

## 🔍 TensorBoard Monitoring

View training progress in real-time:

```bash
# View all experiments
tensorboard --logdir trained_models/experiments/

# View specific experiment
tensorboard --logdir trained_models/experiments/experiment_1/ratio_10_80_10/set_1/tensorboard

# View specific port
tensorboard --logdir trained_models/experiments/ --port 6006
```

Then open browser to: `http://localhost:6006`

### Available in TensorBoard:

- **Loss**: Training and validation loss curves
- **Metrics**: 
  - Accuracy, Precision, Recall, Specificity
  - F1 Score, MCC, AUC, Balanced Accuracy
- **ConfusionMatrix**: Visual confusion matrices per epoch
- **Scalars**: All metrics tracked over training

## 💡 Example Workflow

### Step 1: Check Configuration

```python
from config import BATCH_SIZE, LR, EPOCHS, HIDDEN_DIMS
print(f"Training with {BATCH_SIZE} batch size for {EPOCHS} epochs")
print(f"Hidden layers: {HIDDEN_DIMS}")
```

### Step 2: Train Model

```bash
# Simple training with defaults
python train.py

# Or from notebook
from train import main
import argparse
args = argparse.Namespace(
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=50,
    patience=7,
    dropout=0.3,
    device='cuda',
    seed=42,
    data_dir=DATA_DIR,
    save_root=SAVE_ROOT,
    exp_num=None,
    ratio=None,
    set=None
)
main(args)
```

### Step 3: Analyze Results

```python
import json
import pandas as pd

# Load experiment results
exp_dir = "trained_models/experiments/experiment_1"
results_df = []

for ratio_folder in os.listdir(exp_dir):
    for set_name in os.listdir(os.path.join(exp_dir, ratio_folder)):
        results_file = os.path.join(exp_dir, ratio_folder, set_name, "results.json")
        if os.path.exists(results_file):
            with open(results_file) as f:
                data = json.load(f)
                results_df.append({
                    'ratio': ratio_folder,
                    'set': set_name,
                    **data['best_metrics']
                })

df = pd.DataFrame(results_df)
print(df.describe())
```

### Step 4: View in TensorBoard

```bash
tensorboard --logdir trained_models/experiments/experiment_1
# Open http://localhost:6006 in browser
```

## 🔧 Advanced Configuration

### Modify Hyperparameters

Edit `config.py`:

```python
# Custom settings
BATCH_SIZE = 32
LR = 5e-4
WEIGHT_DECAY = 5e-5
EPOCHS = 100
PATIENCE = 15
DROPOUT = 0.4
HIDDEN_DIMS = [256, 128]  # Smaller model
```

Then train:
```bash
python train.py
```

### Training Custom Model Variants

```bash
# Conservative training (high regularization)
python train.py --batch-size 32 --lr 1e-4 --weight-decay 1e-3 --dropout 0.5

# Aggressive training (lower regularization)
python train.py --batch-size 128 --lr 5e-3 --weight-decay 1e-5 --dropout 0.1

# Long training with early stopping
python train.py --epochs 200 --patience 20
```

## 📋 File Descriptions

### `config.py`
Centralized configuration file with all hyperparameters, paths, and settings.

### `dataset.py`
PyTorch Dataset class that loads pre-computed embeddings from `.pt` files.

### `model.py`
SpliceSiteClassifier model definition with configurable architecture.

### `metrics.py`
Metric calculations using torchmetrics:
- Creates metrics collection for batch updates
- Computes final metrics from predictions
- Calculates specificity, AUC, balanced accuracy

### `train_set.py`
Core training loop with:
- One-epoch training/evaluation
- Gradient clipping for stability
- Confusion matrix logging to TensorBoard
- Metrics tracking and model checkpointing

### `train.py`
Main entry point with:
- CLI argument parsing
- Experiment directory management
- Configuration snapshot saving
- Model architecture summary generation
- Training orchestration

### `cm_visualize.py`
Visualization utilities for confusion matrices.

### `main.ipynb`
Jupyter notebook interface for:
- Configuration viewing
- Training execution
- Results visualization
- TensorBoard monitoring instructions

## 🎓 Best Practices

1. **Always check config before training**: Review `config.py` to ensure correct settings
2. **Use experiment numbers**: Let the system auto-increment experiment numbers
3. **Monitor TensorBoard early**: Start TensorBoard before training to watch progress
4. **Save experiment notes**: Record in `args.json` what each experiment tests
5. **Compare experiments**: Use `results.json` files to compare across experiments
6. **Reproducibility**: Always set `--seed` for reproducible results

## 🐛 Troubleshooting

### GPU Out of Memory
```bash
# Reduce batch size
python train.py --batch-size 32
```

### Training Too Slow
```bash
# Increase batch size (if VRAM allows)
python train.py --batch-size 128
```

### Model Overfitting
```bash
# Increase regularization
python train.py --weight-decay 5e-4 --dropout 0.5 --patience 5
```

### No Improvement
```bash
# Adjust learning rate
python train.py --lr 5e-4  # Lower
# or
python train.py --lr 5e-3  # Higher
```

## 📝 Citation & References

This task implements splicing site prediction using pre-trained DNA sequence embeddings and modern deep learning best practices including:
- Gradient clipping for training stability
- L2 regularization (weight decay)
- Comprehensive metrics from torchmetrics
- Experiment tracking and reproducibility
- TensorBoard visualization

## 📞 Support

For issues or questions:
1. Check TensorBoard logs for training curves
2. Review `results.json` for metrics
3. Check `model_summary.txt` for architecture details
4. Verify config in `config.json`

---

**Last Updated**: January 2026  
**Task**: Splicing Site Prediction  
