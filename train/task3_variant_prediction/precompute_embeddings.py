import os
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from config import (
    TRAIN_PARQUET,
    VAL_PARQUET,
    TEST_PARQUET,
    EMB_DIR,
    TRAIN_EMB,
    VAL_EMB,
    TEST_EMB,
    NT_MODEL,
    ESM_MODEL,
    DNA_BATCH,
    PROT_BATCH,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def center_embedding(hidden, attention_mask):
    # hidden: [B, L, H]; attention_mask: [B, L]
    embs = []
    for i in range(hidden.size(0)):
        seq_len = int(attention_mask[i].sum().item())
        center = seq_len // 2
        embs.append(hidden[i, center, :].cpu())
    return torch.stack(embs)


def embed_sequences(seqs, tokenizer, model, batch_size):
    all_embs = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), batch_size)):
            batch = seqs[i : i + batch_size]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=None,
            )
            input_ids = tokens["input_ids"].to(DEVICE)
            attention_mask = tokens["attention_mask"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            embs = center_embedding(hidden, attention_mask)
            all_embs.append(embs)
    return torch.cat(all_embs, dim=0)


def process_split(parquet_path, out_path, dna_tokenizer, dna_model, prot_tokenizer, prot_model):
    df = pd.read_parquet(parquet_path)
    print(f"Embedding split {parquet_path} ({len(df)} rows)")

    dna_ref = df["ref_seq"].astype(str).tolist()
    dna_alt = df["alt_seq"].astype(str).tolist()
    prot_ref = df["prot_ref_seq"].astype(str).tolist()
    prot_alt = df["prot_alt_seq"].astype(str).tolist()

    dna_ref_emb = embed_sequences(dna_ref, dna_tokenizer, dna_model, DNA_BATCH)
    dna_alt_emb = embed_sequences(dna_alt, dna_tokenizer, dna_model, DNA_BATCH)
    prot_ref_emb = embed_sequences(prot_ref, prot_tokenizer, prot_model, PROT_BATCH)
    prot_alt_emb = embed_sequences(prot_alt, prot_tokenizer, prot_model, PROT_BATCH)
    labels = torch.tensor(df["label"].values, dtype=torch.long)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {
            "dna_ref": dna_ref_emb,
            "dna_alt": dna_alt_emb,
            "prot_ref": prot_ref_emb,
            "prot_alt": prot_alt_emb,
            "label": labels,
        },
        out_path,
    )
    print(f"[saved] {out_path}")


def main():
    os.makedirs(EMB_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    print("Loading NT...")
    dna_tokenizer = AutoTokenizer.from_pretrained(NT_MODEL)
    dna_model = AutoModel.from_pretrained(NT_MODEL).to(DEVICE)

    print("Loading ESM-2...")
    prot_tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
    prot_model = AutoModel.from_pretrained(ESM_MODEL).to(DEVICE)

    process_split(TRAIN_PARQUET, TRAIN_EMB, dna_tokenizer, dna_model, prot_tokenizer, prot_model)
    process_split(VAL_PARQUET, VAL_EMB, dna_tokenizer, dna_model, prot_tokenizer, prot_model)
    process_split(TEST_PARQUET, TEST_EMB, dna_tokenizer, dna_model, prot_tokenizer, prot_model)

    del dna_model, prot_model
    torch.cuda.empty_cache()


