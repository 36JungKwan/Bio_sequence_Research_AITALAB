import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torchinfo import summary

from metrics import create_metrics_collection, compute_metrics_batch, compute_metrics, get_confusion_matrix
from cm_visualize import save_confusion_matrix

def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_confusion_matrix(cm_array, epoch, stage, num_classes):
    """Create and return confusion matrix figure."""
    class_names = [f"Class {i}" for i in range(num_classes)]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix - {stage} - Epoch {epoch}')
    plt.tight_layout()
    return fig


def train_one_epoch(model, loader, optimizer, criterion, device, metrics_collection):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    metrics_collection['accuracy'].reset()

    pbar = tqdm(loader, desc="Training", leave=False)
    for emb, label in pbar:
        emb, label = emb.to(device), label.to(device)

        optimizer.zero_grad()
        logits = model(emb)
        loss = criterion(logits, label)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item() * len(emb)
        
        # Update metrics
        compute_metrics_batch(logits, label, metrics_collection)
        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(loader.dataset)
    metrics_values = {k: v.compute().item() if k != 'cm' else v.compute() 
                      for k, v in metrics_collection.items()}
    
    return avg_loss, metrics_values


def eval_one_epoch(model, loader, criterion, device, metrics_collection, writer=None, epoch=0, stage="val"):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []

    for metric in metrics_collection.values():
        if hasattr(metric, 'reset'):
            metric.reset()

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for emb, label in pbar:
            emb, label = emb.to(device), label.to(device)
            logits = model(emb)
            loss = criterion(logits, label)
            total_loss += loss.item() * len(emb)

            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            
            # Update torchmetrics
            compute_metrics_batch(logits, label, metrics_collection)

    avg_loss = total_loss / len(loader.dataset)
    
    # Compute final metrics
    metrics_values = {}
    for k, v in metrics_collection.items():
        if k != 'cm':
            metrics_values[k] = v.compute().item()
    
    # Legacy metrics computation for compatibility
    legacy_metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics_values.update(legacy_metrics)
    
    cm = get_confusion_matrix(all_labels, all_preds)
    
    # Log confusion matrix to TensorBoard
    if writer is not None:
        cm_fig = plot_confusion_matrix(cm, epoch, stage, num_classes=cm.shape[0])
        writer.add_figure(f"ConfusionMatrix/{stage}", cm_fig, epoch)
        plt.close(cm_fig)

    return avg_loss, metrics_values, cm, all_labels, all_preds, all_probs


def train_model(
    model,
    train_ds,
    val_ds,
    save_dir,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=50,
    patience=7,
    device='cuda',
    seed=42
):
    """Train model with experiment tracking."""
    seed_everything(seed)
    
    os.makedirs(save_dir, exist_ok=True)
    tb_dir = os.path.join(save_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Create metrics collection
    num_classes = 3  # Splicing site classes
    metrics_collection = create_metrics_collection(num_classes, device)

    best_val_loss = float("inf")
    patience_counter = 0
    best_metrics = None
    best_cm = None
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'metrics': []
    }

    print(f"Training model: {save_dir}")
    
    for epoch in range(1, max_epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, metrics_collection
        )
        
        for metric in metrics_collection.values():
            if hasattr(metric, 'reset'):
                metric.reset()
        
        val_loss, val_metrics, cm, all_labels, all_preds, all_probs = eval_one_epoch(
            model, val_loader, criterion, device, metrics_collection,
            writer=writer, epoch=epoch, stage="val"
        )

        # Log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        
        for metric_name, metric_val in val_metrics.items():
            if metric_name != 'cm' and isinstance(metric_val, (int, float)):
                writer.add_scalar(f"Metrics/{metric_name}", metric_val, epoch)

        # Print progress
        metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in val_metrics.items() if k != 'cm' and isinstance(v, (int, float))])
        print(f"[Epoch {epoch}] Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | {metrics_str}")

        # Track history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['metrics'].append(val_metrics)

        # Early stopping and checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            best_cm = cm
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            print(f"  → New best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Save confusion matrix
    if best_cm is not None:
        save_confusion_matrix(best_cm, os.path.join(save_dir, "confusion_matrix.png"))
    else:
        print("Warning: confusion matrix is None → skip save.")

    writer.close()
    
    return best_metrics, best_cm, training_history