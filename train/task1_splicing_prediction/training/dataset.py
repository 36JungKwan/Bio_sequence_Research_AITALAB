import torch
from torch.utils.data import Dataset

class EmbDataset(Dataset):
    def __init__(self, pt_file):
        data = torch.load(pt_file)
        self.emb = data["embeddings"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, idx):
        return self.emb[idx], self.labels[idx]
