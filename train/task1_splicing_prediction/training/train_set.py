import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from metrics import compute_metrics, get_confusion_matrix
from cm_visualize import save_confusion_matrix

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 50
PATIENCE = 7
DEVICE = "cuda"

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for emb, label in loader:
        emb, label = emb.to(device), label.to(device)

        optimizer.zero_grad()
        logits = model(emb)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(emb)

    return total_loss / len(loader.dataset)


def eval_one_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for emb, label in loader:
            emb, label = emb.to(device), label.to(device)
            logits = model(emb)

            loss = criterion(logits, label)
            total_loss += loss.item() * len(emb)

            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    cm = get_confusion_matrix(all_labels, all_preds)

    return total_loss / len(loader.dataset), metrics, cm


def train_model(
    model,
    train_ds,
    val_ds,
    save_dir,
    batch_size=BATCH_SIZE,
    lr=LR,
    max_epochs=EPOCHS,
    patience=PATIENCE,
    device=DEVICE
):

    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_metrics = None
    best_cm = None

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, metrics, cm = eval_one_epoch(model, val_loader, criterion, device)

        for key, value in metrics.items():
            writer.add_scalar(f"Metrics/{key}", value, epoch)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        print(f"[Epoch {epoch}] Train={train_loss:.4f} | Val={val_loss:.4f} | "
              + " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics
            best_cm = cm
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping.")
            break
    
    if best_cm is not None:
        save_confusion_matrix(best_cm, os.path.join(save_dir, "confusion_matrix.png"))
    else:
        print("Warning: confusion matrix is None → skip save.")

    writer.close()
    return best_metrics, best_cm