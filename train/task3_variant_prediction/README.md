# Task 3: Variant-Level Pathogenicity Prediction

Multi-modal fusion model để dự đoán tính gây bệnh (Pathogenicity) của biến thể gen sử dụng embedding từ Nucleotide Transformer (NT) và ESM-2.

## 📋 Tổng quan

Pipeline này thực hiện:
1. **Data Preparation**: Split ClinVar variants theo chromosome (chr20/21 làm test)
2. **Embedding Extraction**: Trích embedding từ NT (DNA) và ESM-2 (Protein) - **zero-shot** (chưa fine-tune)
3. **Multi-modal Fusion**: Kết hợp DNA + Protein embeddings để dự đoán Pathogenic vs Benign
4. **Experiment Tracking**: Tự động lưu config, results, và model checkpoints cho mỗi experiment

## 🏗️ Kiến trúc Model

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

## 📁 Cấu trúc Thư mục

```
task3_variant_prediction/
├── config.py                    # Cấu hình hyperparameters
├── split_data.py                # Split parquet theo chromosome
├── precompute_embeddings.py     # Trích embedding NT + ESM-2
├── dataset.py                  # PyTorch Dataset loader
├── model.py                     # FusionClassifier model
├── train.py                     # Training script với experiment tracking
├── main.ipynb                   # Notebook để chạy toàn bộ pipeline
├── README.md                    # File này
│
├── data/                        # Split data (tự động tạo)
│   ├── train.parquet
│   ├── val.parquet
│   └── test.parquet
│
├── embeddings/                   # Precomputed embeddings (tự động tạo)
│   ├── train_embeddings.pt
│   ├── val_embeddings.pt
│   ├── test_embeddings.pt
│   ├── best_fusion_model.pt     # Global best model
│   └── experiments/             # Tất cả experiments
│       ├── baseline_v1/
│       │   ├── config.json      # Config snapshot
│       │   ├── config.py         # Copy của config.py
│       │   ├── args.json         # Arguments đã dùng
│       │   ├── results.json      # Test results
│       │   ├── best_model.pt     # Model checkpoint
│       │   └── tensorboard/      # TensorBoard logs
│       └── ...
│
└── runs/                        # TensorBoard logs
    ├── baseline_v1/
    └── ...
```

## 🚀 Cài đặt

### Yêu cầu

- Python 3.8+
- PyTorch (với CUDA nếu có GPU)
- Transformers (HuggingFace)
- Các packages khác: `pandas`, `numpy`, `torchmetrics`, `tensorboard`, `tqdm`, `seaborn`, `matplotlib`

### Cài đặt dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install pandas numpy torchmetrics tensorboard tqdm seaborn matplotlib
```

## 📖 Hướng dẫn Sử dụng

### Cách 1: Sử dụng Notebook (Khuyến nghị)

1. **Mở `main.ipynb`** và chạy tuần tự các cells:

   **Cell 1-2: Split Data**
   ```python
   from split_data import main as split_main
   from config import RAW_PARQUET
   split_main(RAW_PARQUET)
   ```
   - Lọc chỉ giữ `Pathogenic` và `Benign` variants
   - Split: chr20/21 → test, còn lại → train/val (15% val)

   **Cell 3-4: Precompute Embeddings**
   ```python
   %env TOKENIZERS_PARALLELISM=false
   from precompute_embeddings import main as emb_main
   emb_main()
   ```
   - Trích embedding từ NT (DNA) và ESM-2 (Protein)
   - Lưu vào `.pt` files (có thể mất vài phút/giờ tùy GPU)

   **Cell 5-6: Train Model**
   ```python
   from train import train
   from config import LR, EPOCHS, BATCH_SIZE, PATIENCE, DROPOUT, SEED
   import argparse

   parser = argparse.ArgumentParser()
   parser.add_argument("--lr", type=float, default=LR)
   parser.add_argument("--epochs", type=int, default=EPOCHS)
   parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
   parser.add_argument("--patience", type=int, default=PATIENCE)
   parser.add_argument("--dropout", type=float, default=DROPOUT)
   parser.add_argument("--seed", type=int, default=SEED)
   parser.add_argument("--exp_name", type=str, default="baseline_v1")
   parser.add_argument("--log_dir", type=str, default=None)

   args = parser.parse_args([])
   result = train(args)
   ```

   **Cell 7-8: Xem lại Experiments**
   - Xem danh sách tất cả experiments
   - Xem chi tiết một experiment cụ thể

### Cách 2: Sử dụng Command Line

```bash
# 1. Split data
python split_data.py --parquet <path_to_parquet>

# 2. Precompute embeddings
python precompute_embeddings.py

# 3. Train với exp_name cụ thể
python train.py --exp_name baseline_v1 --lr 1e-3 --dropout 0.2

# 4. Train với exp_name khác, override config
python train.py --exp_name experiment_v2 --lr 5e-4 --dropout 0.3 --patience 3
```

## 🎯 Các Tính năng Chính

### 1. **Auto Experiment Tracking**

Mỗi lần train, hệ thống tự động:
- Tạo thư mục `embeddings/experiments/<exp_name>/`
- Lưu config snapshot (`config.json`)
- Lưu arguments đã dùng (`args.json`)
- Lưu test results (`results.json`)
- Copy model checkpoint (`best_model.pt`)
- Copy TensorBoard logs

### 2. **Experiment Naming**

- **Có thể đặt tên**: `--exp_name baseline_v1`
- **Tự động tạo**: Nếu không có, tạo theo timestamp `exp_20241201_143022`

### 3. **Configuration Management**

- Tất cả hyperparameters trong `config.py`
- Có thể override từ command line hoặc notebook
- Mỗi experiment lưu snapshot config để reproduce

### 4. **TensorBoard Integration**

- Tự động log: loss, metrics, confusion matrices
- HPARAMS tab: So sánh hyperparameters và metrics giữa experiments
- Xem: `tensorboard --logdir runs`

### 5. **Multi-modal Fusion**

- **DNA Branch**: NT embedding cho `ref_seq` và `alt_seq` (601bp, center token)
- **Protein Branch**: ESM-2 embedding cho `prot_ref_seq` và `prot_alt_seq` (101aa, center token)
- **Fusion**: `[E_ref, E_alt, E_alt - E_ref]` per modality → Concatenate → MLP

## 📊 Xem lại Kết quả

### 1. Trong Notebook

**Xem danh sách tất cả experiments:**
```python
# Cell 8 trong main.ipynb
# Hiển thị bảng: exp_name | timestamp | test_auc | test_acc | test_mcc | ...
```

**Xem chi tiết một experiment:**
```python
# Cell 9 trong main.ipynb
exp_name = "baseline_v1"  # Thay bằng exp_name bạn muốn
# Hiển thị config và results chi tiết
```

### 2. TensorBoard

```bash
# Xem tất cả experiments
tensorboard --logdir train/task3_variant_prediction/runs

# Trong TensorBoard:
# - SCALARS: Loss/metrics curves
# - HPARAMS: So sánh hyperparameters và metrics
# - IMAGES: Confusion matrices
```

### 3. Đọc File JSON

```python
import json

# Đọc config
with open("embeddings/experiments/baseline_v1/config.json", "r") as f:
    config = json.load(f)

# Đọc results
with open("embeddings/experiments/baseline_v1/results.json", "r") as f:
    results = json.load(f)
```

## 🔧 Cấu hình

Chỉnh sửa `config.py` để thay đổi:

```python
# Models
NT_MODEL = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
ESM_MODEL = "facebook/esm2_t33_650M_UR50D"

# Sequence lengths
DNA_SEQ_LEN = 601
PROT_SEQ_LEN = 101

# Training hyperparameters
PROJ_DIM = 512
FUSION_HIDDEN = [512, 256]
DROPOUT = 0.2
LR = 1e-3
EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 128

# Data split
TEST_CHROMS = {"chr20", "chr21", "20", "21"}
VAL_RATIO = 0.15
```

## 📝 Ví dụ Workflow

### Experiment 1: Baseline
```python
parser.add_argument("--exp_name", type=str, default="baseline_v1")
# Kết quả: AUC=0.9850, Acc=0.9397
```

### Experiment 2: Tăng Dropout (giảm overfitting)
```python
parser.add_argument("--exp_name", type=str, default="baseline_v2_dropout03")
parser.add_argument("--dropout", type=float, default=0.3)
# So sánh với baseline_v1
```

### Experiment 3: Giảm Learning Rate
```python
parser.add_argument("--exp_name", type=str, default="baseline_v3_lr5e4")
parser.add_argument("--lr", type=float, default=5e-4)
# So sánh với các experiments trước
```

### So sánh trong TensorBoard:
```bash
tensorboard --logdir runs
# Tab HPARAMS → Chọn experiments → Parallel coordinates plot
```

## 📈 Metrics

Model được đánh giá bằng:
- **AUC** (Area Under ROC Curve)
- **Accuracy**
- **MCC** (Matthews Correlation Coefficient)
- **F1 Score** (macro/micro)
- **Confusion Matrix**

## ⚠️ Lưu ý

1. **GPU Memory**: ESM-2 t33_650M cần ~16GB+ VRAM. Nếu thiếu, giảm `PROT_BATCH` trong `config.py`
2. **Precompute Time**: Trích embedding có thể mất vài giờ tùy số lượng variants và GPU
3. **Overfitting**: Nếu thấy train loss giảm nhưng val loss tăng → tăng dropout, giảm LR, hoặc thêm regularization
4. **Data Path**: Cập nhật `RAW_PARQUET` trong `config.py` trước khi chạy

## 🐛 Troubleshooting

### Lỗi: Out of Memory
- Giảm `BATCH_SIZE` trong `config.py`
- Giảm `PROT_BATCH` (cho ESM-2)
- Giảm `DNA_BATCH` (cho NT)

### Lỗi: File not found
- Kiểm tra `RAW_PARQUET` path trong `config.py`
- Đảm bảo đã chạy `split_data.py` trước `precompute_embeddings.py`

### Lỗi: CUDA out of memory
- Giảm batch sizes
- Dùng CPU: `device = "cpu"` (sẽ chậm hơn nhiều)

## 📚 Tài liệu Tham khảo

- **Nucleotide Transformer**: [InstaDeepAI/nucleotide-transformer](https://huggingface.co/InstaDeepAI/nucleotide-transformer-500m-human-ref)
- **ESM-2**: [facebook/esm2](https://huggingface.co/facebook/esm2_t33_650M_UR50D)
- **ClinVar**: [NCBI ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/)

## 📄 License

Xem LICENSE file trong repository chính.

---

**Tác giả**: Bio Sequence Research Team  
