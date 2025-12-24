import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from config import (
    RAW_PARQUET,
    DATA_DIR,
    TRAIN_PARQUET,
    VAL_PARQUET,
    TEST_PARQUET,
    TEST_CHROMS,
    VAL_RATIO,
    SEED,
)


def filter_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Pathogenic (1) and Benign (0)."""
    keep = ["Pathogenic", "Benign"]
    df = df[df["ClinicalSignificance"].isin(keep)].copy()
    label_map = {"Pathogenic": 1, "Benign": 0}
    df["label"] = df["ClinicalSignificance"].map(label_map)
    return df


def normalize_chrom(val: str) -> str:
    s = str(val)
    if s.lower().startswith("chr"):
        return s.lower()
    return f"chr{s}"


def split(df: pd.DataFrame):
    df = df.copy()
    df["CHROM_norm"] = df["CHROM"].apply(normalize_chrom)

    test_mask = df["CHROM_norm"].isin({c if c.startswith("chr") else f"chr{c}" for c in TEST_CHROMS})
    test_df = df[test_mask]
    trainval_df = df[~test_mask]

    train_df, val_df = train_test_split(
        trainval_df,
        test_size=VAL_RATIO,
        random_state=SEED,
        stratify=trainval_df["label"],
    )
    return train_df, val_df, test_df


def main(parquet_path: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.read_parquet(parquet_path)
    required = ["CHROM", "ref_seq", "alt_seq", "prot_ref_seq", "prot_alt_seq", "ClinicalSignificance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = filter_labels(df)
    train_df, val_df, test_df = split(df)

    train_df.to_parquet(TRAIN_PARQUET, index=False)
    val_df.to_parquet(VAL_PARQUET, index=False)
    test_df.to_parquet(TEST_PARQUET, index=False)

    print("Saved:")
    print(f"  train: {len(train_df)} -> {TRAIN_PARQUET}")
    print(f"  val  : {len(val_df)} -> {VAL_PARQUET}")
    print(f"  test : {len(test_df)} -> {TEST_PARQUET}")


