import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score, average_precision_score, top_k_accuracy_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize

def compute_metrics(labels, preds, probs=None, k=2):
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
    n_classes = 3 if probs is None else probs.shape[1]
    
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    
    # Specificity
    cm = confusion_matrix(labels, preds)
    specificities = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)
    mean_specificity = np.mean(specificities)

    # AUC và AUPRC, Top-K
    auc = 0.0
    auprc = 0.0
    top_k_acc = 0.0
    
    if probs is not None:
        probs_array = np.array(probs)
        try:
            # AUC One-vs-Rest
            auc = roc_auc_score(labels, probs_array, multi_class='ovr', average='macro')
            
            # AUPRC (Phải binarize nhãn để tính từng lớp sau đó lấy average macro)
            labels_bin = label_binarize(labels, classes=range(n_classes))
            if n_classes == 2 and probs_array.shape[1] > 1:
                # Trường hợp đặc biệt nếu chỉ có 2 lớp
                auprc = average_precision_score(labels, probs_array[:, 1])
            else:
                auprc = average_precision_score(labels_bin, probs_array, average='macro')
                
            # Top-K Accuracy
            top_k_acc = top_k_accuracy_score(labels, probs_array, k=k, labels=range(n_classes))
            
        except Exception as e:
            print(f"Warning: Could not compute probability metrics: {e}")

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'specificity': mean_specificity,
        'auc': auc,
        'auprc': auprc, 
        'top_k_acc': top_k_acc, 
        'balanced_acc': bal_acc,
    }

def get_confusion_matrix(labels, preds):
    """Get confusion matrix from predictions."""
    return confusion_matrix(labels, preds)
