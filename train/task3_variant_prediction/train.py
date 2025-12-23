import os
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from dataset import VariantEmbDataset
from model import FusionClassifier
from config import (
    TRAIN_EMB,
    VAL_EMB,
    TEST_EMB,
    PROJ_DIM,
    FUSION_HIDDEN,
    DROPOUT,
    LR,
    EPOCHS,
    PATIENCE,
    BATCH_SIZE,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_metrics(labels, logits):
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    labels_np = torch.tensor(labels).numpy()
    auc = roc_auc_score(labels_np, probs)
    f1 = f1_score(labels_np, preds)
    acc = accuracy_score(labels_np, preds)
    return {"auc": auc, "f1": f1, "acc": acc}


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    total_loss = 0.0
    all_labels, all_logits = [], []
    if is_train:
        model.train()
    else:
        model.eval()

    for dna_ref, dna_alt, prot_ref, prot_alt, label in loader:
        dna_ref = dna_ref.to(DEVICE)
        dna_alt = dna_alt.to(DEVICE)
        prot_ref = prot_ref.to(DEVICE)
        prot_alt = prot_alt.to(DEVICE)
        label = label.to(DEVICE).float()

        with torch.set_grad_enabled(is_train):
            logits = model(dna_ref, dna_alt, prot_ref, prot_alt)
            loss = criterion(logits, label)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(label)
        all_labels.append(label.detach().cpu())
        all_logits.append(logits.detach().cpu())

    total = len(loader.dataset)
    avg_loss = total_loss / total
    labels_cat = torch.cat(all_labels)
    logits_cat = torch.cat(all_logits)
    metrics = compute_metrics(labels_cat, logits_cat)
    return avg_loss, metrics


def train():
    print(f"Device: {DEVICE}")
    train_ds = VariantEmbDataset(TRAIN_EMB)
    val_ds = VariantEmbDataset(VAL_EMB)
    test_ds = VariantEmbDataset(TEST_EMB)

    dna_dim = train_ds.dna_ref.shape[1]
    prot_dim = train_ds.prot_ref.shape[1]

    model = FusionClassifier(
        dna_dim=dna_dim,
        prot_dim=prot_dim,
        proj_dim=PROJ_DIM,
        hidden_dims=FUSION_HIDDEN,
        dropout=DROPOUT,
    ).to(DEVICE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    patience = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, optimizer=None)

        log_line = (
            f"[{epoch}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={val_metrics['auc']:.4f} val_f1={val_metrics['f1']:.4f} val_acc={val_metrics['acc']:.4f}"
        )
        print(log_line)

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            best_state = model.state_dict()
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_metrics = run_epoch(model, test_loader, criterion, optimizer=None)
    print(f"[TEST] loss={test_loss:.4f} auc={test_metrics['auc']:.4f} f1={test_metrics['f1']:.4f} acc={test_metrics['acc']:.4f}")

    torch.save(model.state_dict(), os.path.join(os.path.dirname(TRAIN_EMB), "fusion_model.pt"))
    return {"test_loss": test_loss, **test_metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fusion classifier on precomputed embeddings.")
    _ = parser.parse_args()
    train()

