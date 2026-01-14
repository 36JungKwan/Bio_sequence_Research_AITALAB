# HyenaDNA Variant Prediction Project

## Mô tả

Project này sử dụng mô hình **HyenaDNA** để dự đoán tính chất của biến thể DNA (pathogenic/benign). Code được tổ chức trong Jupyter notebook với các module riêng biệt cho config, dataset, model, embedding và training.

## Cấu trúc Code

`HyenaDNA.ipynb` chứa các cell sau:

### 1. Config File (Cell 1)
Định nghĩa các đường dẫn dữ liệu và tham số:
- Đường dẫn đến data files (parquet)
- Đường dẫn lưu embeddings
- Cấu hình model HyenaDNA và hyperparameters

### 2. Dataset (Cell 3)
- **`HyenaDNADataset`**: Load embeddings đã precompute từ file `.pt`
- Trả về `(dna_ref, dna_alt, label)` cho mỗi sample

### 3. Model (Cell 5)
- **`ModalityProjector`**: Project DNA embeddings `[ref, alt, diff]` → `proj_dim`
- **`DNAClassifier`**: MLP classifier sử dụng DNA embeddings từ HyenaDNA

### 4. Precompute Embeddings (Cell 7)
- **`embed_dna_hyenadna()`**: Embed DNA sequences bằng HyenaDNA model
- **`process_split()`**: Xử lý train/val/test splits
- Lưu embeddings vào file `.pt` để tái sử dụng

### 5. Training Script (Cell 9)
- Training loop với early stopping
- Evaluation metrics: AUC, Accuracy, MCC, Precision, Recall, Specificity
- TensorBoard logging và confusion matrix visualization
- Lưu best model và results

## Yêu cầu Dependencies

pip install torch transformers pandas numpy torchmetrics matplotlib seaborn tqdm torchinfo tensorboard
