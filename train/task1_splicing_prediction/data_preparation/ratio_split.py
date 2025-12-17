import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def stratified_sample(df, group_col, n):
    if n >= len(df):
        return df.sample(len(df), replace=False)

    samples = []
    grouped = df.groupby(group_col)

    for chrom, subdf in grouped:
        k = int(len(subdf) / len(df) * n)
        k = min(k, len(subdf))
        if k > 0:
            samples.append(subdf.sample(k, replace=False))
    
    out = pd.concat(samples, ignore_index=True)

    if len(out) < n:
        extra = df.drop(out.index).sample(n - len(out), replace=False)
        out = pd.concat([out, extra], ignore_index=True)

    return out


def create_ratios_keep_all_pos(df, splice_col, chrom_col):
    neg = df[df[splice_col] == 0].copy()
    one = df[df[splice_col] == 1].copy()
    two = df[df[splice_col] == 2].copy()

    print("Available:\n Class 0:", len(neg), "\nClass 1,", len(one), "\nClass 2,", len(two))

    ratios = {
        "10_1_1": (10, 1, 1),
        "4_1_1":  (4, 1, 1),
        "2_1_1":  (2, 1, 1),
        "1_1_1":  (1, 1, 1)
    }
    output_sets = {}

    n_one = len(one)
    n_two = len(two)

    # Unit = max(positive_class_1, positive_class_2)
    unit = max(n_one, n_two)

    print(f"unit = {unit}")

    for tag, (r_neg, _, _) in ratios.items():
        print(f"\nGenerating ratio {tag} ...")

        n_neg_per_set = unit * r_neg
        total_needed_neg = n_neg_per_set * 10  # 10 sets, no overlapping

        if total_needed_neg > len(neg):
            raise ValueError(
                f"Not enough negative samples for ratio {tag}. "
                f"Need {total_needed_neg}, have {len(neg)}"
            )

        # Shuffle negative
        neg_shuffled = neg.sample(len(neg), replace=False, random_state=42).reset_index(drop=True)
        neg_blocks = np.array_split(neg_shuffled.iloc[:total_needed_neg], 10)

        sets = []

        for i in range(10):
            print(f"  - Creating dataset {i+1}/10")

            # Negative block i
            neg_block = neg_blocks[i]

            # Positive stratified (ensuring chrom distribution)
            acc_sample = stratified_sample(one, chrom_col, n_one)
            don_sample = stratified_sample(two, chrom_col, n_two)

            # Negative stratified chrom-based
            neg_sample = stratified_sample(neg_block, chrom_col, n_neg_per_set)

            # Combine
            df_out = pd.concat([acc_sample, don_sample, neg_sample], ignore_index=True)
            df_out = df_out.sample(len(df_out), replace=False).reset_index(drop=True)

            sets.append(df_out)

        output_sets[tag] = sets

    return output_sets


def stratified_train_val_split(df, chrom_col, val_size):
    # if any chromosome has only 1 record -> singleton -> add into train
    value_counts = df[chrom_col].value_counts()
    singletons = df[df[chrom_col].isin(value_counts[value_counts == 1].index)]
    df_multi = df[df[chrom_col].isin(value_counts[value_counts > 1].index)]

    train_df, val_df = train_test_split(
        df_multi,
        test_size=val_size,
        stratify=df_multi[chrom_col],
        random_state=42
    )

    # Singleton 
    if len(singletons) > 0:
        train_df = pd.concat([train_df, singletons], ignore_index=True)

    # Shuffle
    train_df = train_df.sample(len(train_df), replace=False).reset_index(drop=True)
    val_df = val_df.sample(len(val_df), replace=False).reset_index(drop=True)
    return train_df, val_df


def export_train_val_sets(gs_sets, base_folder, chrom_col, val_size):
    if not os.path.exists(base_folder):
        os.makedirs(base_folder, exist_ok=True)

    for ratio_tag, list_of_dfs in gs_sets.items():
        print(f"\n=== Exporting ratio {ratio_tag} ({len(list_of_dfs)} sets) ===")

        ratio_folder = f"{base_folder}/{ratio_tag}"
        os.makedirs(ratio_folder, exist_ok=True)

        for i, df_set in enumerate(list_of_dfs, start=1):

            set_folder = f"{ratio_folder}/set_{i}"
            os.makedirs(set_folder, exist_ok=True)

            train_df, val_df = stratified_train_val_split(df_set, chrom_col, val_size)

            train_path = f"{set_folder}/train.csv"
            val_path   = f"{set_folder}/val.csv"

            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)

            print(f"  - Saved: {train_path}  ({len(train_df)} rows)")
            print(f"  - Saved: {val_path}    ({len(val_df)} rows)")

    print("\nDONE.")
            

def ratio_splitting(df, chrom_col, splice_col, val_size):
    gs_sets = create_ratios_keep_all_pos(df, splice_col=splice_col, chrom_col=chrom_col)
    export_train_val_sets(gs_sets=gs_sets, base_folder="train_val", chrom_col=chrom_col, val_size=val_size)
