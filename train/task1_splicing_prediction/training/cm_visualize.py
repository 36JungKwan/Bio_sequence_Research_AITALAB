import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def save_confusion_matrix(cm, save_path, class_names=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=True,
                yticklabels=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
