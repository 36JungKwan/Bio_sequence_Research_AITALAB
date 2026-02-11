import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import từ các module của bạn
from dataset import EmbDataset
from model import SpliceSiteClassifier
from config import HIDDEN_DIMS, DROPOUT, NUM_CLASSES
from metrics import compute_metrics, get_confusion_matrix

def run_ensemble_strategy(strategy_name, ratios_to_include, test_files, data_dir, exp_dir, device):
    """
    Thực hiện Soft Voting Ensemble cho 1 chiến lược cụ thể.
    """
    # Tận dụng strategy_name để in log rõ ràng
    print(f"\n" + "="*50)
    print(f"🚀 STARTING ENSEMBLE STRATEGY: {strategy_name}")
    print(f"📦 Including Ratios: {ratios_to_include}")
    print("="*50)

    # Gom danh sách model dựa trên các ratio được chỉ định
    model_paths = []
    for r in ratios_to_include:
        for s in range(1, 11):
            p = os.path.join(exp_dir, r, f"set_{s}", "best_model.pt")
            if os.path.exists(p): 
                model_paths.append(p)
    
    if not model_paths:
        print(f"⚠️ [Warning] No models found for {strategy_name} in {exp_dir}")
        return None

    print(f"✅ Found {len(model_paths)} models. Starting inference...")

    results = {}
    for test_csv in test_files:
        test_tag = test_csv.replace('.csv', '')
        test_pt = os.path.join(data_dir, test_csv.replace(".csv", "_embeddings.pt"))
        
        if not os.path.exists(test_pt):
            print(f"❓ [Skip] {test_tag} embeddings not found.")
            continue

        print(f"  -> Testing on: {test_tag}...", end=" ", flush=True)

        # Logic inference (giữ nguyên như cũ)
        ds = EmbDataset(test_pt)
        loader = DataLoader(ds, batch_size=128, shuffle=False)
        data_pt = torch.load(test_pt, map_location='cpu')
        all_labels = data_pt['labels'].numpy()
        embedding_dim = data_pt['embeddings'].shape[1]

        ensemble_probs = np.zeros((len(ds), NUM_CLASSES))

        for m_path in model_paths:
            model = SpliceSiteClassifier(embedding_dim, HIDDEN_DIMS, DROPOUT, NUM_CLASSES).to(device)
            model.load_state_dict(torch.load(m_path, map_location=device))
            model.eval()

            batch_probs = []
            with torch.no_grad():
                for emb, _ in loader:
                    logits = model(emb.to(device))
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    batch_probs.append(probs)
            
            ensemble_probs += np.concatenate(batch_probs, axis=0)
            del model # Giải phóng VRAM ngay lập tức

        ensemble_probs /= len(model_paths)
        ensemble_preds = np.argmax(ensemble_probs, axis=-1)

        metrics = compute_metrics(all_labels.tolist(), ensemble_preds.tolist(), ensemble_probs.tolist())
        cm = get_confusion_matrix(all_labels.tolist(), ensemble_preds.tolist())

        # Lưu kết quả kèm theo tên chiến lược để nhận diện
        results[test_tag] = {
            "strategy": strategy_name,
            "metrics": metrics, 
            "cm": cm
        }
        print("Done!")
    
    return results