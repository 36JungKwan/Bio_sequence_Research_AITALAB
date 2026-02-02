import os
import json
import argparse
import random
import shutil
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from torchinfo import summary

from dataset import EmbDataset
from train_set import train_model, seed_everything
from model import SpliceSiteClassifier
from config import (
    DATA_DIR,
    EXPERIMENTS_DIR,
    BATCH_SIZE,
    LR,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    HIDDEN_DIMS,
    DROPOUT,
    NUM_CLASSES,
    SEED,
)


def get_next_experiment_number(experiments_dir):
    """Get the next experiment number."""
    if not os.path.exists(experiments_dir):
        return 1
    
    exp_dirs = [d for d in os.listdir(experiments_dir) if d.startswith('experiment_')]
    if not exp_dirs:
        return 1
    
    numbers = []
    for d in exp_dirs:
        try:
            num = int(d.replace('experiment_', ''))
            numbers.append(num)
        except ValueError:
            continue
    
    return max(numbers) + 1 if numbers else 1


def create_experiment_dir(base_dir, experiment_num):
    """Create experiment directory structure."""
    exp_dir = os.path.join(base_dir, f"experiment_{experiment_num}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_experiment_config(exp_dir, args, config_dict):
    """Save experiment configuration and arguments."""
    # Save args as JSON
    args_dict = vars(args)
    with open(os.path.join(exp_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=2)
    
    # Save config as JSON
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Copy config.py to experiment folder for reproducibility
    config_py_path = os.path.join(os.path.dirname(__file__), "config.py")
    if os.path.exists(config_py_path):
        shutil.copy(config_py_path, os.path.join(exp_dir, "config.py"))
        print(f"  ✅ Config.py copied to {exp_dir}/config.py")


def save_results(exp_dir, metrics, training_history):
    """Save training results."""
    results = {
        'best_metrics': {k: float(v) if isinstance(v, (int, float)) else str(v) 
                        for k, v in metrics.items()},
        'training_history': {
            'train_loss': [float(x) for x in training_history['train_loss']],
            'val_loss': [float(x) for x in training_history['val_loss']],
        }
    }
    
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


def train_splicing_model(
    ratio_folder,
    set_name,
    data_dir,
    experiment_dir,
    batch_size=BATCH_SIZE,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    epochs=EPOCHS,
    patience=PATIENCE,
    device='cuda'
):
    """Train model for a specific dataset split."""
    
    # Load datasets
    train_pt = os.path.join(data_dir, ratio_folder, set_name, "train_embeddings.pt")
    val_pt = os.path.join(data_dir, ratio_folder, set_name, "val_embeddings.pt")
    
    if not os.path.exists(train_pt) or not os.path.exists(val_pt):
        print(f"  ⚠️  Missing embeddings for {ratio_folder}/{set_name}")
        return None
    
    train_ds = EmbDataset(train_pt)
    val_ds = EmbDataset(val_pt)
    
    # Initialize model
    embedding_dim = train_ds.emb.shape[1]
    model = SpliceSiteClassifier(
        embedding_dim=embedding_dim,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES
    ).to(device)
    
    # Create save directory for this experiment
    save_dir = os.path.join(experiment_dir, ratio_folder, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n  Training {ratio_folder}/{set_name}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Save dir: {save_dir}")
    
    # Print model summary
    print(f"\n  Model Architecture:")
    print(f"  {'-'*50}")
    try:
        model_summary = summary(
            model,
            input_size=(batch_size, embedding_dim),
            col_names=["input_size", "output_size", "num_params"],
            device=device,
            verbose=0
        )
        print(model_summary)
        
        # Save model summary to file
        summary_path = os.path.join(save_dir, "model_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(str(model_summary))
        print(f"  ✅ Model summary saved to {summary_path}")
    except Exception as e:
        print(f"  ⚠️  Could not generate model summary: {e}")
    print(f"  {'-'*50}\n")
    
    # Train
    best_metrics, best_cm, training_history = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        save_dir=save_dir,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=epochs,
        patience=patience,
        device=device,
        seed=SEED
    )
    
    return {
        'metrics': best_metrics,
        'cm': best_cm,
        'history': training_history,
        'ratio': ratio_folder,
        'set': set_name
    }


def main(args):    
    # Set seeds
    seed_everything(args.seed)
    
    # Create experiment directory
    os.makedirs(args.save_root, exist_ok=True)
    exp_num = args.exp_num if args.exp_num is not None else get_next_experiment_number(args.save_root)
    exp_dir = create_experiment_dir(args.save_root, exp_num)
    
    # Print detailed configuration
    print(f"\n{'='*70}")
    print(f"EXPERIMENT {exp_num}: SPLICING PREDICTION TRAINING")
    print(f"{'='*70}")
    print(f"TRAINING CONFIGURATION:")
    print(f"{'='*70}")
    print(f"  Timestamp:           {datetime.now().isoformat()}")
    print(f"  Experiment dir:      {exp_dir}")
    print(f"  Data dir:            {args.data_dir}")
    print(f"  Device:              {args.device}")
    print(f"  Seed:                {args.seed}")
    print(f"  Batch size:          {args.batch_size}")
    print(f"  Learning rate:       {args.lr}")
    print(f"  Weight decay:        {args.weight_decay}")
    print(f"  Epochs:              {args.epochs}")
    print(f"  Patience:            {args.patience}")
    print(f"  Dropout:             {args.dropout}")
    print(f"  Hidden dims:         {HIDDEN_DIMS}")
    print(f"  Num classes:         {NUM_CLASSES}")
    print(f"  Embedding dim:       auto-detected")
    print(f"{'='*70}\n")
    
    # Save configuration
    config_dict = {
        'timestamp': datetime.now().isoformat(),
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'patience': args.patience,
        'dropout': args.dropout,
        'hidden_dims': HIDDEN_DIMS,
        'num_classes': NUM_CLASSES,
        'embedding_dim': 'auto',
        'device': args.device,
        'seed': args.seed,
    }
    save_experiment_config(exp_dir, args, config_dict)
    print(f"Configuration saved to {exp_dir}/config.json")
    print(f"Arguments saved to {exp_dir}/args.json\n")
    
    # Train models
    all_results = []
    
    if not os.path.exists(args.data_dir):
        print(f"Data directory not found: {args.data_dir}")
        return
    
    ratio_folders = sorted(os.listdir(args.data_dir))
    if args.ratio:
        ratio_folders = [r for r in ratio_folders if r == args.ratio]
    
    for ratio_folder in ratio_folders:
        ratio_path = os.path.join(args.data_dir, ratio_folder)
        if not os.path.isdir(ratio_path):
            continue
        
        print(f"\nProcessing {ratio_folder}:")
        
        set_folders = sorted(os.listdir(ratio_path))
        if args.set:
            set_folders = [s for s in set_folders if s == args.set]
        
        for set_name in set_folders:
            set_path = os.path.join(ratio_path, set_name)
            if not os.path.isdir(set_path):
                continue
            
            result = train_splicing_model(
                ratio_folder=ratio_folder,
                set_name=set_name,
                data_dir=args.data_dir,
                experiment_dir=exp_dir,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                device=args.device
            )
            
            if result:
                all_results.append(result)
                # Save individual results
                save_results(
                    os.path.join(exp_dir, ratio_folder, set_name),
                    result['metrics'],
                    result['history']
                )
    
    # Summary
    print(f"\n{'='*70}")
    print(f"EXPERIMENT {exp_num} COMPLETED!")
    print(f"{'='*70}")
    print(f"Results saved to: {exp_dir}")
    print(f"Models saved in each set directory")
    print(f"Metrics and results saved as JSON")
    print(f"Model summaries saved as .txt files")
    print(f"Confusion matrices visualized in TensorBoard")
    print(f"\nTo view TensorBoard logs:")
    print(f"   tensorboard --logdir {os.path.join(exp_dir, 'ratio_*', 'set_*', 'tensorboard')}")
    print(f"{'='*70}\n")