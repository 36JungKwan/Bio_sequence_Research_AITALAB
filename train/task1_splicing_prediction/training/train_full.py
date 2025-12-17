import os
import torch
from dataset import EmbDataset
from train_set import train_model
from model import SpliceSiteClassifier
from fileio import ResultTable


def train_all(base_dir, save_root, device):

    results = ResultTable()

    for ratio_folder in sorted(os.listdir(base_dir)):
        ratio_path = os.path.join(base_dir, ratio_folder)
        if not os.path.isdir(ratio_path):
            continue

        print(f"\n===== RATIO {ratio_folder} =====")

        for set_name in sorted(os.listdir(ratio_path)):
            set_path = os.path.join(ratio_path, set_name)
            if not os.path.isdir(set_path):
                continue

            train_pt = os.path.join(set_path, "train_embeddings.pt")
            val_pt = os.path.join(set_path, "val_embeddings.pt")

            if not os.path.exists(train_pt) or not os.path.exists(val_pt):
                continue

            print(f"\n--- Training {ratio_folder}/{set_name} ---")

            train_ds = EmbDataset(train_pt)
            val_ds = EmbDataset(val_pt)

            model = SpliceSiteClassifier(
                embedding_dim=train_ds.emb.shape[1],
                hidden_dims=[512, 256],
                dropout=0.3,
                num_classes=3
            ).to(device)

            save_dir = os.path.join(save_root, ratio_folder, set_name)

            metrics, cm = train_model(
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                save_dir=save_dir,
                device=device
            )

            results.add(ratio_folder, set_name, metrics)

    results.to_csv(os.path.join(save_root, "summary_results.csv"))
    print("\nSaved summary_results.csv")

    return results.to_df()