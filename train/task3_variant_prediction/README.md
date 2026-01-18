# Task 3: Variant-Level Pathogenicity Prediction

Multi-modal fusion model to predict pathogenicity of genetic variants using embeddings from Nucleotide Transformer (NT) and ESM-2.

## 📋 Overview

This pipeline performs:
1. **Data Preparation**: Split ClinVar variants by chromosome (chr20/21 as test set)
2. **Embedding Extraction**: Extract embeddings from NT (DNA) and ESM-2 (Protein) (pre-trained, not fine-tuned)
3. **Multi-modal Fusion**: Combine DNA + Protein embeddings to predict Pathogenic vs Benign
4. **Experiment Tracking**: Automatically save config, results, and model checkpoints for each experiment

## 🏗️ Model Architecture

```
Input Variants (ClinVar)
    ↓
[DNA Sequences] → NT (zero-shot) → E_dna_ref, E_dna_alt
[Protein Sequences] → ESM-2 (zero-shot) → E_prot_ref, E_prot_alt
    ↓
Fusion Layer: [E_ref, E_alt, E_alt - E_ref] per modality
    ↓
Concatenate DNA + Protein → MLP Classifier
    ↓
Pathogenicity Score (Pathogenic=1, Benign=0)
```

## 📁 Directory Structure

```
task3_variant_prediction/
├── config.py                    # Hyperparameter configuration
├── split_data.py                # Split parquet by chromosome
├── precompute_embeddings.py     # Extract NT + ESM-2 embeddings
├── dataset.py                  # PyTorch Dataset loader
├── model.py                     # FusionClassifier model
├── train.py                     # Training script with experiment tracking
├── main.ipynb                   # Notebook to run full pipeline
├── README.md                    # This file
│
├── data/                        # Split data (auto-generated)
│   ├── train.parquet
│   ├── val.parquet
│   └── test.parquet
│
├── embeddings/                   # Precomputed embeddings (auto-generated)
│   ├── train_embeddings.pt
│   ├── val_embeddings.pt
│   ├── test_embeddings.pt
│   ├── best_fusion_model.pt     # Global best model
│   └── experiments/             # All experiments
│       ├── experiment_1/
│       │   ├── config.json      # Config snapshot
│       │   ├── config.py         # Copy of config.py
│       │   ├── args.json         # Arguments used
│       │   ├── results.json      # Test results
│       │   ├── best_model.pt     # Model checkpoint
│       │   └── tensorboard/      # TensorBoard logs
│       └── ...
│
└── runs/                        # TensorBoard logs
    ├── experiment_1/
    └── ...
```

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch (with CUDA if GPU available)
- Transformers (HuggingFace)
- Other packages: `pandas`, `numpy`, `torchmetrics`, `tensorboard`, `tqdm`, `seaborn`, `matplotlib`

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install pandas numpy torchmetrics tensorboard tqdm seaborn matplotlib
```

## 📖 Usage Guide

### Method 1: Using Notebook (Recommended)

1. **Open `main.ipynb`** and run cells sequentially:

   **Cell 1-2: Split Data**
   ```python
   from split_data import main as split_main
   from config import RAW_PARQUET
   split_main(RAW_PARQUET)
   ```
   - Filter to keep only `Pathogenic` and `Benign` variants
   - Split: chr20/21 → test, rest → train/val (15% val)

   **Cell 3-4: Precompute Embeddings**
   ```python
   %env TOKENIZERS_PARALLELISM=false
   from precompute_embeddings import main as emb_main
   emb_main()
   ```
   - Extract embeddings from NT (DNA) and ESM-2 (Protein)
   - Save to `.pt` files (may take several minutes/hours depending on GPU)

   **Cell 5-6: Train Model**
   ```python
   from train import train
   import argparse

   parser = argparse.ArgumentParser()
   
   # Fusion & modality options
   parser.add_argument("--mode", type=str, default="both", choices=['dna', 'prot', 'both'])
   parser.add_argument("--fusion_method", type=str, default="concat", choices=['concat', 'cross_attn'])
   parser.add_argument("--feature_mode", type=str, default="all", choices=['all', 'ref_alt', 'diff', 'ref', 'alt'])
   parser.add_argument("--use_gating", type=bool, default=False, help="Enable gated fusion")
   
   # Hyperparameters
   parser.add_argument("--lr", type=float, default=1e-3)
   parser.add_argument("--epochs", type=int, default=30)
   parser.add_argument("--batch_size", type=int, default=128)
   parser.add_argument("--dropout", type=float, default=0.2)
   parser.add_argument("--weight_decay", type=float, default=1e-4)
   
   # Experiment setup
   parser.add_argument("--exp_name", type=str, default="experiment_1")
   parser.add_argument("--seed", type=int, default=42)
   
   args = parser.parse_args([])
   result = train(args)
   ```

   **Cell 7-8: View Experiments**
   - List all experiments
   - View details of specific experiment

### Method 2: Using Command Line

```bash
# 1. Split data
python split_data.py

# 2. Precompute embeddings
python precompute_embeddings.py

# 3. Train with default config (concat fusion, all features)
python train.py --exp_name experiment_1 --lr 1e-3 --dropout 0.2

# 4. Train with cross-attention fusion
python train.py --exp_name experiment_2_cross_attn --fusion_method cross_attn --lr 1e-3

# 5. Train with gated fusion (adaptive modality weighting)
python train.py --exp_name experiment_3_gated --use_gating True --lr 1e-3

# 6. Train with diff features only
python train.py --exp_name experiment_4_diff --feature_mode diff --lr 1e-3

# 7. Train DNA-only (ablation)
python train.py --exp_name experiment_5_dna_only --mode dna --lr 1e-3

# 8. Train Protein-only (ablation)
python train.py --exp_name experiment_6_prot_only --mode prot --lr 1e-3

# 9. Advanced: Cross-attention + gated fusion + diff features
python train.py --exp_name experiment_7_advanced \
  --fusion_method cross_attn \
  --use_gating True \
  --feature_mode diff \
  --lr 5e-4 \
  --dropout 0.3
```

## 🎯 Key Features

### 1. **Flexible Fusion Modes**

Choose how to combine modalities:
- **`mode='both'`** (default): Use both DNA and Protein embeddings
- **`mode='dna'`**: DNA-only predictions
- **`mode='prot'`**: Protein-only predictions

Useful for ablation studies to understand modality contributions.

### 2. **Multiple Fusion Methods**

Control how modalities are combined:
- **`fusion_method='concat'`** (default): Simple concatenation → MLP
- **`fusion_method='cross_attn'`**: Cross-attention blocks to capture modality interactions
  - DNA attends to Protein features
  - Protein attends to DNA features
  - Better captures synergistic effects

### 3. **Gated Fusion Mechanism** (NEW)

**Adaptive modality weighting via learned gates:**
- **`use_gating=True`**: Combine modalities using a learned gating mechanism
- **`use_gating=False`**: Use simple concatenation (default)

**Gating Formula:**
```
combined = [DNA_feat || Protein_feat]
g = Sigmoid(Linear(combined))           # Gate weight ∈ [0, 1]
fused = g * DNA_feat + (1 - g) * Protein_feat
```

**Key Benefits:**
- Learns sample-specific modality weights (not fixed)
- g≈1: Trusts DNA more (useful for nucleotide-driven variants)
- g≈0.5: Balanced trust in both modalities
- g≈0: Trusts Protein more (useful for protein-consequence variants)
- More parameter-efficient than concatenation (512 vs 1024 input dims to MLP)
- Automatically learns which modality is more important per sample

**Gate Analysis & Interpretability:**
The model automatically generates:
- Gate value distribution histogram showing modality preferences
- Identifies "DNA-dependent" variants (g > 0.7)
- Identifies "Protein-dependent" variants (g < 0.3)
- Traces samples to original parquet with gate weights for downstream analysis

### 4. **Feature Mode Options**

Flexible feature combinations for each modality:
- **`feature_mode='all'`** (default): `[E_ref, E_alt, E_alt - E_ref]` → full information
- **`feature_mode='ref_alt'`**: `[E_ref, E_alt]` → reference and alternative
- **`feature_mode='diff'`**: `E_alt - E_ref` → only change/difference
- **`feature_mode='ref'`**: `E_ref` → only reference
- **`feature_mode='alt'`**: `E_alt` → only alternative

### 4. **Auto Experiment Tracking**

Each training run automatically:
- Create directory `embeddings/experiments/<exp_name>/`
- Save config snapshot (`config.json`)
- Save used arguments (`args.json`)
- Save test results (`results.json`)
- Copy model checkpoint (`best_model.pt`)
- Copy TensorBoard logs

### 5. **Comprehensive Metrics Tracking**

Real-time monitoring with TensorBoard:
- **Loss curves**: Training vs Validation
- **Classification metrics**: Accuracy, Precision, Recall, F1, MCC, AUC
- **Confusion matrices**: Visualized per epoch
- **Gradient norms**: Model training stability
- **HPARAMS dashboard**: Compare experiments with different hyperparameters

### 6. **Multi-modal Fusion with Cross-Attention**

Advanced fusion mechanism:
```
DNA Embedding (proj_dim=512)
    ↓
Cross-Attention Block: DNA attends to Protein
    ↓
Protein Embedding (proj_dim=512)
    ↓
Cross-Attention Block: Protein attends to DNA
    ↓
Concatenate → MLP Classifier
```

### 7. **Modality Projector with Feature Engineering**

Each modality uses a `ModalityProjector`:
- Projects embeddings to shared dimension space
- Supports different feature combinations (ref/alt/diff)
- LayerNorm + GELU activation for better representation

## 📊 View Results

### 1. In Notebook

**View list of all experiments:**
```python
# Cell 8 in main.ipynb
# Display table: exp_name | timestamp | test_auc | test_acc | test_mcc | ...
```

**View details of specific experiment:**
```python
# Cell 9 in main.ipynb
exp_name = "experiment_1"  # Change to your exp_name
# Display config and detailed results
```

### 2. TensorBoard

```bash
# View all experiments
tensorboard --logdir train/task3_variant_prediction/runs

# In TensorBoard:
# - SCALARS: Loss/metrics curves
# - HPARAMS: Compare hyperparameters and metrics
# - IMAGES: Confusion matrices
```

### 3. Read JSON Files

```python
import json

# Read config
with open("embeddings/experiments/experiment_1/config.json", "r") as f:
    config = json.load(f)

# Read results
with open("embeddings/experiments/experiment_1/results.json", "r") as f:
    results = json.load(f)
```

## 🔧 Configuration

Edit `config.py` to set defaults:

```python
# ============ MODALITY OPTIONS ============
MODE = 'both'               # Options: 'dna', 'prot', 'both'
FUSION_METHOD = 'concat'    # Options: 'concat', 'cross_attn'
FEATURE_MODE = 'all'        # Options: 'all', 'ref_alt', 'diff', 'ref', 'alt'
USE_GATING = False          # Enable/disable gated fusion

# ============ MODELS ============
NT_MODEL = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
ESM_MODEL = "facebook/esm2_t33_650M_UR50D"

# ============ ARCHITECTURE ============
PROJ_DIM = 512              # Projection dimension for each modality
FUSION_HIDDEN = [512, 256]  # MLP hidden layer sizes
DROPOUT = 0.2

# ============ TRAINING ============
LR = 1e-3
EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 128
WEIGHT_DECAY = 1e-4         # L2 regularization

# ============ DATA ============
TEST_CHROMS = {"chr20", "chr21"}
VAL_RATIO = 0.15
SEED = 42
```

## 📝 Example Experiments

### Experiment 1: Baseline (Concatenation + All Features)
```bash
python train.py --exp_name exp_baseline --mode both --fusion_method concat --feature_mode all --use_gating False
# Standard approach: concatenate DNA and protein, use all features
```

### Experiment 2: Gated Fusion Only
```bash
python train.py --exp_name exp_gated --mode both --fusion_method concat --use_gating True --lr 1e-3
# Learn adaptive modality weights via gates instead of fixed concatenation
```

### Experiment 3: Cross-Attention Fusion
```bash
python train.py --exp_name exp_cross_attn --mode both --fusion_method cross_attn --feature_mode all --use_gating False --lr 1e-3
# Advanced fusion: DNA and protein attend to each other
```

### Experiment 4: Cross-Attention + Gated Fusion
```bash
python train.py --exp_name exp_cross_attn_gated --mode both --fusion_method cross_attn --use_gating True --lr 1e-3
# Combine cross-attention interaction + gated modality weighting
```

### Experiment 5: Gated Fusion with Difference Features
```bash
python train.py --exp_name exp_gated_diff --mode both --feature_mode diff --use_gating True --lr 1e-3
# Use only change information (E_alt - E_ref) + gated fusion
```

### Experiment 6: Cross-Attention + Gated + Diff Features
```bash
python train.py --exp_name exp_cross_attn_gated_diff \
  --fusion_method cross_attn \
  --use_gating True \
  --feature_mode diff \
  --lr 5e-4 \
  --dropout 0.3 \
  --weight_decay 1e-4
# Full advanced setup: combines all advanced techniques
```

### Experiment 7: DNA-Only (Ablation)
```bash
python train.py --exp_name exp_dna_only --mode dna --lr 1e-3
# Test DNA modality contribution alone
```

### Experiment 8: Protein-Only (Ablation)
```bash
python train.py --exp_name exp_prot_only --mode prot --lr 1e-3
# Test protein modality contribution alone
```

### Compare Experiments in TensorBoard:
```bash
tensorboard --logdir runs
# HPARAMS tab → Select multiple experiments → Parallel coordinates plot
```

## 📈 Advanced Features Reference

### GatingMechanism Class (NEW)

```python
class GatingMechanism(nn.Module):
    """Learn sample-specific modality weighting via sigmoid gates."""
    
    def __init__(self, dim):
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),      # Combine both modalities
            nn.Sigmoid()                   # Output gate ∈ [0, 1]
        )
    
    def forward(self, dna_feat, prot_feat):
        combined = torch.cat([dna_feat, prot_feat], dim=-1)
        g = self.gate(combined)            # [batch, proj_dim]
        fused = g * dna_feat + (1 - g) * prot_feat
        return fused
```

**How it works:**
- Takes concatenated DNA and Protein features
- Learns a gate value per dimension
- Performs element-wise weighted combination
- Each sample gets its own adaptive weights (not fixed)
- More interpretable than concat: can analyze which modality wins per sample

### analyze_gating_behavior() Function (NEW)

Automatically analyzes gate distributions on test set:

```python
def analyze_gating_behavior(model, loader, device):
    # Extract gate values for all test samples
    # Generates histogram of gate distribution
    # Identifies DNA-dependent vs Protein-dependent variants
    # Returns: gate_values, labels, predictions
```

**Output includes:**
1. **Gate Distribution Plot**: Shows how many variants rely on each modality
2. **DNA-Dependent Count**: Number of variants with g > 0.7 (DNA-heavy)
3. **Protein-Dependent Count**: Number of variants with g < 0.3 (Protein-heavy)

### trace_samples_to_parquet() Function (NEW)

Traces analyzed samples back to original data:

```python
def trace_samples_to_parquet(test_parquet_path, all_gates, all_labels, all_preds, threshold=0.6):
    # Merges gate analysis with original variant information
    # Creates interpretable output with: CHROM, POS, REF, ALT, gate_weight, label, correctness
    # Enables downstream analysis of "why did the model make this prediction?"
```

**Output parquet includes:**
- `gate_dna_weight`: How much model trusted DNA (0 to 1)
- `true_label`: Ground truth pathogenicity
- `pred_score`: Predicted probability
- `is_correct`: Whether prediction matches ground truth
- All original variant info: CHROM, POS, REF, ALT, sequences

### FusionClassifier with Gating

In forward pass, when `use_gating=True`:
```python
# After cross-attention (if enabled)
dna_f, prot_f = ...  # [batch, proj_dim] each

# Gated fusion
combined = torch.cat([dna_f, prot_f], dim=-1)
g = self.gater.gate(combined)  # [batch, proj_dim]
fused = g * dna_f + (1 - g) * prot_f

logits = self.classifier(fused)

# Return gates for analysis
if return_gates:
    return logits, g.mean(dim=-1)  # Average gate across dimensions
```

## 📈 Metrics

Model is evaluated using:
- **AUC** (Area Under ROC Curve)
- **Accuracy**
- **MCC** (Matthews Correlation Coefficient)
- **F1 Score** (macro/micro)
- **Confusion Matrix**

## ⚠️ Important Notes

1. **GPU Memory**: ESM-2 t33_650M requires ~16GB+ VRAM. If memory limited, reduce `PROT_BATCH` in `config.py`
2. **Precompute Time**: Extracting embeddings may take several hours depending on variant count and GPU
3. **Overfitting**: If train loss decreases but val loss increases → increase dropout, decrease LR, or add regularization
4. **Data Path**: Update `RAW_PARQUET` in `config.py` before running

## 🐛 Troubleshooting

### Error: Out of Memory
- Reduce `BATCH_SIZE` in `config.py`
- Reduce `PROT_BATCH` (for ESM-2)
- Reduce `DNA_BATCH` (for NT)

### Error: File not found
- Check `RAW_PARQUET` path in `config.py`
- Ensure `split_data.py` runs before `precompute_embeddings.py`

### Error: CUDA out of memory
- Reduce batch sizes
- Use CPU: `device = "cpu"` (will be much slower)

## 📚 Advanced Features Reference

### ModalityProjector Class

```python
class ModalityProjector(nn.Module):
    """Projects embeddings to shared dimension with flexible feature modes."""
    
    def __init__(self, emb_dim, proj_dim, dropout, feature_mode='all'):
        # Input dimension depends on feature_mode:
        # - 'all': emb_dim*3 (ref, alt, diff)
        # - 'ref_alt': emb_dim*2 (ref, alt)
        # - 'diff'/'ref'/'alt': emb_dim
```

Feature engineering per modality:
- Input: ref & alt embeddings from NT or ESM-2
- Output: projected feature vector
- Supports different combinations for flexible ablations

### CrossAttentionBlock

```python
class CrossAttentionBlock(nn.Module):
    """Multi-head attention for modality interaction."""
    
    def __init__(self, dim, num_heads=4, dropout=0.1):
        # Captures interactions between modalities
        # Used in cross_attn fusion method
```

Benefits:
- Learns modality-specific representations
- Captures synergistic effects between DNA and protein
- More expressive than simple concatenation

### FusionClassifier Architecture

**Concat Mode:**
```
[DNA features]     → Projector → [512]
                                   ├→ Concat → [1024] → MLP → Output
[Protein features] → Projector → [512]
                                    
```

**Cross-Attention Mode:**
```
[DNA features]     → Projector → [512]
                                    ├→ Cross-Attn (DNA attends to Prot)
[Protein features] → Projector → [512]     ├→ Concat → [1024] → MLP → Output
                                    ├→ Cross-Attn (Prot attends to DNA)
                    
```

## Advanced Model Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | 'both' | 'dna', 'prot', or 'both' - which modalities to use |
| `fusion_method` | str | 'concat' | 'concat' or 'cross_attn' - how to combine modalities |
| `feature_mode` | str | 'all' | 'all', 'ref_alt', 'diff', 'ref', 'alt' - feature combination || `use_gating` | bool | False | Enable gated fusion for adaptive modality weighting || `proj_dim` | int | 512 | Projection dimension for each modality |
| `fusion_hidden` | list | [512, 256] | MLP hidden layer dimensions |
| `dropout` | float | 0.2 | Dropout rate (0.0-1.0) |
| `weight_decay` | float | 1e-4 | L2 regularization strength |
| `lr` | float | 1e-3 | Learning rate |
| `batch_size` | int | 128 | Training batch size |

## 📚 References

- **Nucleotide Transformer**: [InstaDeepAI/nucleotide-transformer](https://huggingface.co/InstaDeepAI/nucleotide-transformer-500m-human-ref)
- **ESM-2**: [facebook/esm2](https://huggingface.co/facebook/esm2_t33_650M_UR50D)
- **ClinVar**: [NCBI ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/)
- **Multi-Head Attention**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
