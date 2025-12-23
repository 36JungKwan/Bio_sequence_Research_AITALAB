import torch
from torch.utils.data import Dataset


class VariantEmbDataset(Dataset):
    def __init__(self, pt_file):
        data = torch.load(pt_file)
        self.dna_ref = data["dna_ref"]
        self.dna_alt = data["dna_alt"]
        self.prot_ref = data["prot_ref"]
        self.prot_alt = data["prot_alt"]
        self.labels = data["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.dna_ref[idx],
            self.dna_alt[idx],
            self.prot_ref[idx],
            self.prot_alt[idx],
            self.labels[idx],
        )

