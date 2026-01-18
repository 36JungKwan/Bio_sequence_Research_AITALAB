import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
import shutil
from datetime import datetime
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchmetrics
from torchmetrics.classification import (
    BinaryAUROC, 
    MulticlassF1Score, 
    BinaryAccuracy,
    MulticlassAccuracy, 
    BinaryMatthewsCorrCoef,
    BinaryConfusionMatrix,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity
)

from dataset import VariantEmbDataset
from model import FusionClassifier
from config import (
    TRAIN_EMB,
    VAL_EMB,
    TEST_EMB,
    TEST_PARQUET,
    PROJ_DIM,
    FUSION_HIDDEN,
    DROPOUT,
    LR,
    EPOCHS,
    PATIENCE,
    BATCH_SIZE,
    SEED,
    MODE,
    FEATURE_MODE,
    FUSION_METHOD,
    USE_GATING
)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_confusion_matrix(cm_tensor, epoch, stage):
    """
    Chuyển đổi tensor confusion matrix thành một hình ảnh matplotlib.
    """
    cm = cm_tensor.cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax,
                xticklabels=['Benign', 'Pathogenic'], 
                yticklabels=['Benign', 'Pathogenic'])
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'Confusion Matrix - {stage} - Epoch {epoch}')
    plt.tight_layout()
    return fig

def analyze_gating_behavior(model, loader, device):
    model.eval()
    all_gates = []
    all_labels = []
    all_preds = []
    
    print("Analyzing gating weights on test set...")
    with torch.no_grad():
        for dna_ref, dna_alt, prot_ref, prot_alt, label in tqdm(loader):
            dna_ref, dna_alt = dna_ref.to(device), dna_alt.to(device)
            prot_ref, prot_alt = prot_ref.to(device), prot_alt.to(device)
            
            # Lấy logits và giá trị gate trung bình
            logits, gate_values = model(dna_ref, dna_alt, prot_ref, prot_alt, return_gates=True)
            
            all_gates.extend(gate_values.cpu().numpy())
            all_labels.extend(label.numpy())
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())

    all_gates = np.array(all_gates)
    
    # 1. Vẽ phân phối của Gate g
    plt.figure(figsize=(8, 5))
    sns.histplot(all_gates, bins=50, kde=True, color='purple')
    plt.axvline(x=0.5, color='red', linestyle='--')
    plt.title("Phân phối trọng số Gate (g)")
    plt.xlabel("Giá trị g (Càng cao càng tin vào DNA, càng thấp càng tin vào Protein)")
    plt.ylabel("Số lượng mẫu")
    plt.show()

    # 2. In ra các mẫu "DNA-dependent" (g > 0.7)
    dna_heavy_idx = np.where(all_gates > 0.7)[0]
    print(f"Số lượng mẫu phụ thuộc nhiều vào DNA (g > 0.7): {len(dna_heavy_idx)} / {len(all_gates)}")
    
    return all_gates, all_labels, all_preds

def trace_samples_to_parquet(test_parquet_path, all_gates, all_labels, all_preds, threshold=0.6):
    """
    Truy ngược các mẫu có gating weight cao về file parquet gốc.
    """
    # 1. Đọc file parquet gốc của tập test
    df_test = pd.read_parquet(test_parquet_path)
    
    # 2. Tạo DataFrame chứa kết quả phân tích
    analysis_df = pd.DataFrame({
        'gate_dna_weight': all_gates,
        'true_label': all_labels,
        'pred_score': all_preds,
        'is_correct': (np.array(all_labels) == (np.array(all_preds) > 0.5)).astype(int)
    })
    
    # 3. Ghép nối với dữ liệu gốc (đảm bảo thứ tự không thay đổi)
    final_df = pd.concat([df_test.reset_index(drop=True), analysis_df], axis=1)
    
    # 4. Lọc ra các mẫu "DNA-Priority"
    dna_priority_samples = final_df[final_df['gate_dna_weight'] > threshold].copy()
    
    # Sắp xếp theo trọng số giảm dần để xem mẫu nào "DNA nhất"
    dna_priority_samples = dna_priority_samples.sort_values(by='gate_dna_weight', ascending=False)
    
    return dna_priority_samples

def run_epoch(model, loader, criterion, device, metrics_collection, cm_metric, optimizer=None, writer=None, epoch=0, stage="train"):    
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    metrics_collection.reset()
    cm_metric.reset()

    pbar = tqdm(loader, desc=f"{stage.capitalize()} Epoch {epoch}", leave=False)

    for dna_ref, dna_alt, prot_ref, prot_alt, label in pbar:        
        dna_ref = dna_ref.to(device)
        dna_alt = dna_alt.to(device)
        prot_ref = prot_ref.to(device)
        prot_alt = prot_alt.to(device)
        label = label.to(device).float()

        with torch.set_grad_enabled(is_train):
            logits = model(dna_ref, dna_alt, prot_ref, prot_alt)
            loss = criterion(logits, label)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item() * len(label)
        preds = (torch.sigmoid(logits) > 0.5).int()
        metrics_collection.update(preds, label.int())
        cm_metric.update(preds, label.int())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    results = {k: v.item() for k, v in metrics_collection.compute().items()}
    results['loss'] = avg_loss

    if writer:
        for name, value in results.items():
            writer.add_scalar(f"{stage}/{name}", value, epoch)
        
        cm_tensor = cm_metric.compute()
        fig = plot_confusion_matrix(cm_tensor, epoch, stage)
        writer.add_figure(f"ConfusionMatrix/{stage}", fig, epoch)
        plt.close(fig)
            
    return results


def train(args):
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Tự động tạo exp_name nếu không có
    if args.exp_name is None:
        args.exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Tự động tạo log_dir từ exp_name nếu không có
    if args.log_dir is None:
        args.log_dir = os.path.join("runs", args.exp_name)
    
    # Tạo experiment directory
    exp_dir = os.path.join(os.path.dirname(TRAIN_EMB), "experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # In ra configuration
    print("=" * 70)
    print("TRAINING CONFIGURATION:")
    print("=" * 70)
    print(f"  Experiment Name: {args.exp_name}")
    print(f"  Device: {device}")
    print(f"  Mode: {args.mode}")
    print(f"  Fusion Method: {args.fusion_method}")
    print(f"  Use Gating: {args.use_gating}")
    print(f"  Feature Mode: {args.feature_mode}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Patience: {args.patience}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Seed: {args.seed}")
    print(f"  Proj Dim: {args.proj_dim}")
    print(f"  Fusion Hidden: {args.fusion_hidden}")
    print(f"  Log Dir: {args.log_dir}")
    print(f"  Experiment Dir: {exp_dir}")
    print("=" * 70)
    
    # Lưu config snapshot
    config_snapshot = {
        "exp_name": args.exp_name,
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "fusion_method": args.fusion_method, 
        "use_gating": args.use_gating,
        "feature_mode": args.feature_mode,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "dropout": args.dropout,
        "seed": args.seed,
        "proj_dim": args.proj_dim,
        "fusion_hidden": args.fusion_hidden,
        "log_dir": args.log_dir,
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)
    
    # Copy config.py vào experiment folder
    config_py_path = os.path.join(os.path.dirname(__file__), "config.py")
    if os.path.exists(config_py_path):
        shutil.copy(config_py_path, os.path.join(exp_dir, "config.py"))
    
    # Lưu args vào file JSON
    args_dict = vars(args)
    with open(os.path.join(exp_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=2)
    
    writer = SummaryWriter(log_dir=args.log_dir)

    train_ds = VariantEmbDataset(TRAIN_EMB)
    val_ds = VariantEmbDataset(VAL_EMB)
    test_ds = VariantEmbDataset(TEST_EMB)

    loader_args = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    model = FusionClassifier(
        dna_dim=train_ds.dna_ref.shape[1],
        prot_dim=train_ds.prot_ref.shape[1],
        proj_dim=args.proj_dim,
        hidden_dims=args.fusion_hidden,
        dropout=args.dropout,
        mode=args.mode,
        fusion_method=args.fusion_method,
        feature_mode=args.feature_mode,
        use_gating=args.use_gating
    ).to(device)

    print("\n" + "="*30 + " MODEL SUMMARY " + "="*30)
    input_data_shapes = [
        (args.batch_size, train_ds.dna_ref.shape[1]),  # dna_ref
        (args.batch_size, train_ds.dna_ref.shape[1]),  # dna_alt
        (args.batch_size, train_ds.prot_ref.shape[1]), # prot_ref
        (args.batch_size, train_ds.prot_ref.shape[1])  # prot_alt
    ]

    model_stats = summary(
        model, 
        input_size=input_data_shapes,
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        device=device,
        verbose=0
    )
    
    print(model_stats)
    
    summary_path = os.path.join(exp_dir, "model_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(str(model_stats))
    print(f"--> Model summary saved to {summary_path}")
    print("="*75 + "\n")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    metrics = torchmetrics.MetricCollection({
        'auc': BinaryAUROC(),
        'acc': BinaryAccuracy(),
        'mcc': BinaryMatthewsCorrCoef(),
        'balanced_acc': MulticlassAccuracy(num_classes=2, average='macro'),
        'f1_macro': MulticlassF1Score(num_classes=2, average='macro'),
        'precision': BinaryPrecision(),
        'recall': BinaryRecall(),
        'specificity': BinarySpecificity()
    }).to(device)

    cm_metric = BinaryConfusionMatrix().to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    save_path = os.path.join(os.path.dirname(TRAIN_EMB), "best_fusion_model.pt")
    exp_save_path = os.path.join(exp_dir, "best_model.pt")  # Copy vào exp folder

    for epoch in range(1, args.epochs + 1):
        train_res = run_epoch(model, train_loader, criterion, device, metrics, cm_metric, optimizer, writer, epoch, "train")
        val_res = run_epoch(model, val_loader, criterion, device, metrics, cm_metric, None, writer, epoch, "val")

        print(f"[{epoch}] Train Loss: {train_res['loss']:.4f} | Val Loss: {val_res['loss']:.4f} | Train Acc: {train_res['acc']:.4f} | Val Acc: {val_res['acc']:.4f}")

        scheduler.step(val_res['loss'])

        if val_res['loss'] < best_val_loss:
            best_val_loss = val_res['loss']
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            shutil.copy(save_path, exp_save_path)  # Copy vào exp folder
            print(f"--> Saved best model checkpoint to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    print("\n--- Testing with Best Model ---")
    model.load_state_dict(torch.load(save_path))
    test_res = run_epoch(model, test_loader, criterion, device, metrics, cm_metric, None, writer, args.epochs, "test")
    print(f"[TEST] Loss: {test_res['loss']:.4f} | AUC: {test_res['auc']:.4f} | MCC: {test_res['mcc']:.4f} | Acc: {test_res['acc']:.4f} | Spec: {test_res['specificity']:.4f}")
    print(f"[TEST] Balanced Acc: {test_res['balanced_acc']:.4f} | F1_macro: {test_res['f1_macro']:.4f} | Precision: {test_res['precision']:.4f} | Recall: {test_res['recall']:.4f}")

    # ... đoạn code test cuối cùng ...
    print("\n--- Gating Analysis ---")
    gates, labels, preds = analyze_gating_behavior(model, test_loader, device)
    
    # Lưu kết quả analysis vào folder experiment
    analysis_results = {
        "mean_gate_weight": float(np.mean(gates)),
        "dna_priority_count": int(np.sum(gates > 0.5)),
        "protein_priority_count": int(np.sum(gates <= 0.5))
    }
    with open(os.path.join(exp_dir, "gating_analysis.json"), "w") as f:
        json.dump(analysis_results, f, indent=2)

    # Chạy sau khi đã có gates, labels, preds từ hàm phân tích trước
    dna_samples = trace_samples_to_parquet(TEST_PARQUET, gates, labels, preds)

    print("\n--- TOP 5 SAMPLES PRIORITIZING DNA ---")
    # Hiển thị các cột quan trọng (tùy vào các cột trong file parquet của bạn)
    cols_to_show = ['chrom', 'pos', 'ref', 'alt', 'gate_dna_weight', 'true_label', 'is_correct']
    print(dna_samples[cols_to_show].head(5))

    # Lưu lại để làm báo cáo
    dna_samples.to_csv(os.path.join(exp_dir, "dna_priority_cases.csv"), index=False)
    
    # Lưu hparams vào TensorBoard
    hparams = {
        "mode": args.mode,
        "fusion_method": args.fusion_method,
        "use_gating": args.use_gating,
        "feature_mode": args.feature_mode,
        "lr": args.lr,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "proj_dim": args.proj_dim,
        "fusion_hidden": str(args.fusion_hidden),
        "patience": args.patience,
    }
    metrics_dict = {
        "test_auc": test_res['auc'],
        "test_acc": test_res['acc'],
        "test_mcc": test_res['mcc'],
        "test_balanced_acc": test_res['balanced_acc'],
        "test_f1_macro": test_res['f1_macro'],
        "test_precision": test_res['precision'],
        "test_recall": test_res['recall'],
        "test_specificity": test_res['specificity'],
        "test_loss": test_res['loss'],
        "best_val_loss": best_val_loss,
    }
    writer.add_hparams(hparams, metrics_dict)
    
    # Lưu kết quả cuối cùng vào file JSON
    final_results = {
        "exp_name": args.exp_name,
        "timestamp": datetime.now().isoformat(),
        "best_val_loss": float(best_val_loss),
        "test_results": {k: float(v) for k, v in test_res.items()},
        "epochs_trained": epoch,
        "hparams": hparams,
    }
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(final_results, f, indent=2)
    
    # Copy TensorBoard logs vào exp folder (optional)
    if os.path.exists(args.log_dir):
        tb_dest = os.path.join(exp_dir, "tensorboard")
        if not os.path.exists(tb_dest):
            shutil.copytree(args.log_dir, tb_dest)
    
    writer.close()
    
    print(f"\n✓ Experiment saved to: {exp_dir}")
    print(f"  - Config: {os.path.join(exp_dir, 'config.json')}")
    print(f"  - Args: {os.path.join(exp_dir, 'args.json')}")
    print(f"  - Results: {os.path.join(exp_dir, 'results.json')}")
    print(f"  - Model: {exp_save_path}")
    
    return test_res


