import os
import shutil
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

NT_MODEL = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_COLUMN = "sequence"  
LABEL_COLUMN = 'Splicing_types' 
MAX_LENGTH = 128         
BATCH_SIZE = 64

# EMBEDDING EXTRACTION FUNCTION
def extract_embeddings_from_nt(sequences, tokenizer, model, device, batch_size=BATCH_SIZE):
    model.eval()
    all_embeddings = []

    print(f"Extracting embeddings from Nucleotide Transformer... ({len(sequences)} sequences)")
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch = sequences[i:i+batch_size]

            tokens = tokenizer.batch_encode_plus(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            )

            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

            # center embedding
            for j in range(hidden_states.size(0)):
                seq_len = int(attention_mask[j].sum().item())
                center_idx = seq_len // 2
                emb = hidden_states[j, center_idx, :].cpu()
                all_embeddings.append(emb)

    return torch.stack(all_embeddings)

# PROCESS CSV TO EMBEDDING
def process_csv_embedding(csv_path, tokenizer, model, device, replace=False):
    if not os.path.exists(csv_path):
        print(f"[skip] CSV not found: {csv_path}")
        return None

    print(f"\nReading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if SEQ_COLUMN not in df.columns:
        print(f"[skip] Missing column {SEQ_COLUMN}: {csv_path}")
        return None
    if LABEL_COLUMN not in df.columns:
        print(f"[skip] Missing column {LABEL_COLUMN}: {csv_path}")
        return None

    seqs = df[SEQ_COLUMN].astype(str).tolist()
    labels = torch.tensor(df[LABEL_COLUMN].values) # Tensor
    embeddings = extract_embeddings_from_nt(seqs, tokenizer, model, device)

    # Save dictionary
    data_dict = {
        "embeddings": embeddings.cpu().detach(), # Tensor [N, Hidden_Dim]
        "labels": labels.cpu().detach()          # Tensor [N]
    }

    save_path = csv_path.replace(".csv", "_embeddings.pt")
    
    # Try saving with error handling and alternative methods
    try:
        torch.save(data_dict, save_path)
        print(f"[saved] {save_path}")
    except RuntimeError as e:
        print(f"[error] torch.save failed: {e}")
        print(f"[info] Retrying with pickle format...")
        try:
            # Alternative: Save using pickle with protocol 4
            import pickle
            save_path_pkl = save_path.replace(".pt", ".pkl")
            with open(save_path_pkl, 'wb') as f:
                pickle.dump(data_dict, f, protocol=4)
            print(f"[saved] {save_path_pkl}")
            save_path = save_path_pkl
        except Exception as e2:
            print(f"[error] Pickle save also failed: {e2}")
            return None

    # delete original CSV if REPLACE=True
    if replace==True:
        try:
            os.remove(csv_path)
            print(f"[deleted] {csv_path}")
        except Exception as e:
            print(f"[warning] Could not delete CSV: {e}")

    return save_path

# WALK THROUGH train_val/
def embed_train_val_folder(root, replace=False):
    print("...Loading Nucleotide Transformer...")

    tokenizer = AutoTokenizer.from_pretrained(NT_MODEL)
    model = AutoModel.from_pretrained(NT_MODEL).to(DEVICE)
    model.eval()

    print(f"Device = {DEVICE}")
    print(f"Root folder = {root}\n")

    # walk through folder structure
    for parent, dirs, files in os.walk(root):
        if "train.csv" in files or "val.csv" in files:
            print(f"Processing set: {parent}")
            train_csv = os.path.join(parent, "train.csv")
            val_csv = os.path.join(parent, "val.csv")
            # TRAIN
            if os.path.exists(train_csv):
                process_csv_embedding(train_csv, tokenizer, model, DEVICE, replace=replace)
            # VAL
            if os.path.exists(val_csv):
                process_csv_embedding(val_csv, tokenizer, model, DEVICE, replace=replace)
            print(f"Finished embedding set: {parent}\n")

    # cleanup GPU
    del model
    torch.cuda.empty_cache()

    print("[DONE] All embeddings generated.")


def embed_test_folder(root, replace=False):
    """Walk `root` and create embeddings for any `test.csv` files found.

    Saves `*_embeddings.pt` next to each `test.csv`. This keeps test embedding
    generation centralized in this module (call this before running inference).
    """
    print("...Loading Nucleotide Transformer for test embeddings...")

    tokenizer = AutoTokenizer.from_pretrained(NT_MODEL)
    model = AutoModel.from_pretrained(NT_MODEL).to(DEVICE)
    model.eval()

    print(f"Device = {DEVICE}")
    print(f"Root folder = {root}\n")

    for parent, dirs, files in os.walk(root):
        if "test.csv" in files:
            print(f"Processing test set: {parent}")
            test_csv = os.path.join(parent, "test.csv")
            if os.path.exists(test_csv):
                process_csv_embedding(test_csv, tokenizer, model, DEVICE, replace=replace)
            print(f"Finished embedding test set: {parent}\n")

    # cleanup GPU
    del model
    torch.cuda.empty_cache()

    print("[DONE] All test embeddings generated.")