from sklearn.metrics import (
    roc_auc_score, 
    balanced_accuracy_score, 
    matthews_corrcoef,
    accuracy_score,
    f1_score,
    confusion_matrix
)
import numpy as np

def compute_metrics(labels, preds, probs):
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)

    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr")
    except:
        auc = 0.0

    bal_acc = balanced_accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")

    return {
        "auc": auc,
        "balanced_acc": bal_acc,
        "mcc": mcc,
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
    }


def get_confusion_matrix(labels, preds):
    return confusion_matrix(labels, preds)
