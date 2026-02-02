import torch
import numpy as np
from torchmetrics import Accuracy, Precision, Recall, F1Score, MatthewsCorrCoef, ConfusionMatrix, Specificity
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score

def create_metrics_collection(num_classes, device):
    """
    Create a collection of metrics for multi-class classification.
    
    Args:
        num_classes: Number of classes
        device: Device to run metrics on
        
    Returns:
        Dict of metric objects
    """
    return {
        'accuracy': Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(device),
        'precision': Precision(task='multiclass', num_classes=num_classes, average='macro').to(device),
        'recall': Recall(task='multiclass', num_classes=num_classes, average='macro').to(device),
        'specificity': Specificity(task='multiclass', num_classes=num_classes, average='macro').to(device),
        'f1': F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device),
        'mcc': MatthewsCorrCoef(task='multiclass', num_classes=num_classes).to(device),
        'cm': ConfusionMatrix(task='multiclass', num_classes=num_classes).to(device),
    }

def compute_metrics_batch(logits, labels, metrics_dict):
    """
    Update metrics with a batch of predictions.
    
    Args:
        logits: Model output logits [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        metrics_dict: Dictionary of metric objects
    """
    probs = torch.softmax(logits, dim=-1)
    
    for metric_name, metric in metrics_dict.items():
        if metric_name == 'cm':
            metric.update(torch.argmax(probs, dim=-1), labels)
        else:
            metric.update(probs, labels)

def compute_metrics(labels, preds, probs=None):
    """
    Legacy function: compute metrics from predictions.
    
    Args:
        labels: Ground truth labels
        preds: Predicted class labels
        probs: Predicted probabilities
        
    Returns:
        Dict of metric values
    """
    labels = np.array(labels)
    preds = np.array(preds)
    
    from sklearn.metrics import (
        accuracy_score, 
        precision_score, 
        recall_score,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
        balanced_accuracy_score
    )
    
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    
    # Specificity: True Negative Rate (using macro average)
    # For multi-class, we'll compute per-class specificity and average
    cm = confusion_matrix(labels, preds)
    specificities = []
    n_classes = cm.shape[0]
    
    for i in range(n_classes):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)
    
    specificity = np.mean(specificities)
    
    # AUC (One-vs-Rest for multi-class)
    try:
        if probs is not None:
            probs_array = np.array(probs)
            # Ensure probs is properly shaped [n_samples, n_classes]
            if len(probs_array.shape) == 1:
                probs_array = probs_array.reshape(-1, 1)
            auc = roc_auc_score(labels, probs_array, multi_class='ovr')
        else:
            auc = 0.0
    except Exception as e:
        print(f"Warning: Could not compute AUC: {e}")
        auc = 0.0
    
    # Balanced Accuracy
    bal_acc = balanced_accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'specificity': specificity,
        'auc': auc,
        'balanced_acc': bal_acc,
    }

def get_confusion_matrix(labels, preds):
    """Get confusion matrix from predictions."""
    return confusion_matrix(labels, preds)
