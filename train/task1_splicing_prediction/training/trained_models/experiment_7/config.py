import os

# Paths
DATA_DIR = r"D:\Bio_sequence_Research_AITALAB\train\task1_splicing_prediction\data_preparation\train_val"
SAVE_ROOT = os.path.join(os.path.dirname(__file__), "trained_models")
EXPERIMENTS_DIR = os.path.join(SAVE_ROOT, "experiments")

# Training hyperparameters
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 7

# Model architecture
EMBEDDING_DIM = None  # Will be inferred from data
HIDDEN_DIMS = [512, 256]
DROPOUT = 0.3
NUM_CLASSES = 3

# Seeds and reproducibility
SEED = 42
