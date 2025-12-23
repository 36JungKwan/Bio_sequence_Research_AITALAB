import torch
import torch.nn as nn


class ModalityProjector(nn.Module):
    def __init__(self, emb_dim, proj_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 3, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, ref, alt):
        diff = alt - ref
        x = torch.cat([ref, alt, diff], dim=-1)
        return self.net(x)


class FusionClassifier(nn.Module):
    def __init__(self, dna_dim, prot_dim, proj_dim, hidden_dims, dropout):
        super().__init__()
        self.dna_proj = ModalityProjector(dna_dim, proj_dim, dropout)
        self.prot_proj = ModalityProjector(prot_dim, proj_dim, dropout)

        layers = []
        in_dim = proj_dim * 2
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, dna_ref, dna_alt, prot_ref, prot_alt):
        dna_z = self.dna_proj(dna_ref, dna_alt)
        prot_z = self.prot_proj(prot_ref, prot_alt)
        fused = torch.cat([dna_z, prot_z], dim=-1)
        return self.classifier(fused).squeeze(-1)

