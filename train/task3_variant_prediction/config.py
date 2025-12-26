import os

# Paths
RAW_PARQUET = os.environ.get("TASK3_PARQUET", r"D:\Biosequence\variant_protein_sequence_101aa.parquet")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_PARQUET = os.path.join(DATA_DIR, "train.parquet")
VAL_PARQUET = os.path.join(DATA_DIR, "val.parquet")
TEST_PARQUET = os.path.join(DATA_DIR, "test.parquet")

EMB_DIR = os.path.join(os.path.dirname(__file__), "embeddings")
TRAIN_EMB = os.path.join(EMB_DIR, "train_embeddings.pt")
VAL_EMB = os.path.join(EMB_DIR, "val_embeddings.pt")
TEST_EMB = os.path.join(EMB_DIR, "test_embeddings.pt")

# Models
NT_MODEL = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
ESM_MODEL = "facebook/esm2_t33_650M_UR50D"

# Sequence lengths (fixed, center token pooling)
DNA_SEQ_LEN = 601
PROT_SEQ_LEN = 101

# Splits
TEST_CHROMS = {"chr20", "chr21", "20", "21"}
VAL_RATIO = 0.15
SEED = 42

# Embedding + training
DNA_BATCH = 32
PROT_BATCH = 4
PROJ_DIM = 512
FUSION_HIDDEN = [512, 256]
DROPOUT = 0.2
LR = 1e-3
EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 128

