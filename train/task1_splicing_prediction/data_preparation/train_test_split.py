import os
import pandas as pd
from sklearn.model_selection import train_test_split  

def train_test_set_split(file_path, chr_col, test_chr, label_col):
    try:
        df = pd.read_csv(file_path)
        print(f"File: {file_path} ({len(df)} rows)")
    except FileNotFoundError:
        print(f"Error: File not found -> {file_path}")
        exit()

    test_mask = df[chr_col].isin(test_chr)
    test = df[test_mask].copy()
    train = df[~test_mask].copy()

    print("\n--- DONE ---")
    print(f"Train_val set size: {len(train)} rows.")
    print(f"Test set size: {len(test)} rows.")

    print(f"\n--- '{chr_col}' ---")
    print("\nTrain_val set distribution:")
    print(train[chr_col].value_counts(normalize=True).round(4))

    print("\nTest set distribution:")
    print(test[chr_col].value_counts(normalize=True).round(4))

    print(f"\n--- '{label_col}' ---")
    print("Original distribution:")
    print(df[label_col].value_counts(normalize=True).round(4))

    print("\nTrain_val set distribution:")
    print(train[label_col].value_counts(normalize=True).round(4))

    print("\nTest set distribution:")
    print(test[label_col].value_counts(normalize=True).round(4))

    save_folder="raw"
    os.makedirs(save_folder, exist_ok=True)
    train_path = f"{save_folder}/train_val_data.csv"
    train.to_csv(train_path, index=False)
    print(f"\nTrain_val set saved: {train_path}")
    test_path = f"{save_folder}/test_data.csv"
    test.to_csv(test_path, index=False)
    print(f"\nTest set saved: {test_path}")

    return train, test