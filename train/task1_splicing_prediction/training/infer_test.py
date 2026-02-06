import os
import sys
import argparse
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# ensure we can import from sibling data_preparation
THIS_DIR = os.path.dirname(__file__)
DATA_PREP_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'data_preparation'))
if DATA_PREP_DIR not in sys.path:
    sys.path.insert(0, DATA_PREP_DIR)


# ensure training package modules importable
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from dataset import EmbDataset
from model import SpliceSiteClassifier
from config import HIDDEN_DIMS, DROPOUT, NUM_CLASSES
from metrics import compute_metrics, get_confusion_matrix

def run_inference_for_set(data_dir, exp_dir, ratio, set_name, test_data, device, batch_size=64):
    set_data_folder = os.path.join(data_dir, ratio, set_name)
    if not os.path.isdir(set_data_folder):
        print(f"[skip] Data folder missing: {set_data_folder}")
        return None

    test_csv = os.path.join(set_data_folder, test_data)
    test_pt = os.path.join(set_data_folder, test_data.replace(".csv", "_embeddings.pt"))

    # Support a single global test.csv/embeddings placed at data_dir
    used_global = False
    if not os.path.exists(test_pt):
        global_test_pt = os.path.join(data_dir, test_data.replace(".csv", "_embeddings.pt"))
        if os.path.exists(global_test_pt):
            test_pt = global_test_pt
            used_global = True
        else:
            print(f"[skip] No test embeddings for {ratio}/{set_name} at {test_pt}. Please run extract_embed.embed_test_folder first (creates either per-set or global test_embeddings.pt).")
            return None

    # Load embeddings
    data = torch.load(test_pt, map_location='cpu')
    embeddings = data['embeddings']
    labels = data['labels']

    ds = EmbDataset(test_pt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # build model and load best checkpoint
    exp_set_dir = os.path.join(exp_dir, ratio, set_name)
    best_model_path = os.path.join(exp_set_dir, 'best_model.pt')
    if not os.path.exists(best_model_path):
        print(f"[skip] No best_model.pt for {ratio}/{set_name} at {best_model_path}")
        return None

    embedding_dim = embeddings.shape[1]
    model = SpliceSiteClassifier(
        embedding_dim=embedding_dim,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES
    ).to(device)

    state = torch.load(best_model_path, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        # maybe whole model was saved
        try:
            model = state.to(device)
        except Exception as e:
            print(f"[error] Could not load model state for {best_model_path}: {e}")
            return None

    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for emb, label in loader:
            emb = emb.to(device)
            logits = model(emb)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_labels.extend(label.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    cm = get_confusion_matrix(all_labels, all_preds)

    # Save results
    results = {
        'ratio': ratio,
        'set': set_name,
        'metrics': metrics,
        'confusion_matrix': cm.tolist()
    }

    os.makedirs(exp_set_dir, exist_ok=True)
    results_path = os.path.join(exp_set_dir, test_data.replace(".csv", "_results.json"))
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {results_path}")

    return results

