import torch.nn as nn

class SpliceSiteClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dims, dropout, num_classes):
        super().__init__()

        layers = []
        prev_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, embeddings):
        return self.classifier(embeddings)