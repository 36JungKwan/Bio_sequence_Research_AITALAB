import os
import shutil
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

NT_MODEL = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_COLUMN = "ref_seq"  
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
        "embeddings": embeddings.cpu(), # Tensor [N, Hidden_Dim]
        "labels": labels.cpu()          # Tensor [N]
    }

    save_path = csv_path.replace(".csv", "_embeddings.pt")
    torch.save(data_dict, save_path)
    print(f"[saved] {save_path}")

    # delete original CSV if REPLACE=True
    if replace==True:
        os.remove(csv_path)
        print(f"[deleted] {csv_path}")

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