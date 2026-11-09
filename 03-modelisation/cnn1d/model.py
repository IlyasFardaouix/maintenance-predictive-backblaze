"""Architecture du CNN-1D."""
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """3 convolutions 1D le long du temps, puis moyenne et maximum globaux, puis une tête dense.

    Entrée  : (B, WINDOW, n_features)
    Sortie  : (B,) RUL normalisée (RUL / RUL_CAP)
    """

    def __init__(self, n_feat, ch=(64, 64, 128), k=3, dropout=0.3):
        super().__init__()
        p = k // 2
        self.features = nn.Sequential(
            nn.Conv1d(n_feat, ch[0], k, padding=p), nn.BatchNorm1d(ch[0]), nn.ReLU(),
            nn.Conv1d(ch[0], ch[1], k, padding=p), nn.BatchNorm1d(ch[1]), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(ch[1], ch[2], k, padding=p), nn.BatchNorm1d(ch[2]), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * ch[2], 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x):
        h = self.features(x.transpose(1, 2))             # Conv1d attend (B, F, L)
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)   # tendance moyenne + pics
        return self.head(h).squeeze(-1)
