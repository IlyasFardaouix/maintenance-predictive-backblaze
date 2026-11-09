"""Fenêtres glissantes construites à la volée (jamais matérialisées en mémoire).

Une fenêtre est identifiée par l'indice de sa dernière ligne. Elle regroupe les
WINDOW lignes consécutives d'un même segment ; son étiquette est la RUL de cette
dernière ligne.
"""
import numpy as np
import torch

from . import protocole as P


def window_ends(d, window=P.WINDOW):
    """Indices de fin de toutes les fenêtres valides, puis séparés en dégradation / sain."""
    ends = np.flatnonzero(d["pos"] >= window - 1)
    low = ends[d["rul"][ends] < P.RUL_CAP]      # dégradation (RUL < 125)
    high = ends[d["rul"][ends] >= P.RUL_CAP]    # sain (RUL = 125)
    return ends, low, high


def sample_ends(low, high, ratio, rng):
    """Rééquilibrage de l'entraînement : toutes les fenêtres en dégradation
    + `ratio` fois plus de fenêtres saines tirées au hasard."""
    n_h = min(len(high), int(ratio * len(low)))
    e = np.concatenate([low, high[rng.integers(0, len(high), n_h)]])
    rng.shuffle(e)
    return e


class WindowStore:
    """Garde les séries d'un split (sur GPU si possible) et extrait les fenêtres par batch."""

    def __init__(self, d, device, data_device=None, window=P.WINDOW):
        self.device = device
        self.data_device = data_device or device
        self.X = torch.from_numpy(d["X"]).to(self.data_device)
        self.rul = torch.from_numpy(d["rul"]).to(self.data_device)
        self.offsets = torch.arange(-window + 1, 1, device=self.data_device)

    def batch(self, ends_np):
        """Fenêtres (B, WINDOW, n_features) en float32 et cibles RUL normalisées dans [0, 1]."""
        e = torch.from_numpy(np.ascontiguousarray(ends_np)).to(self.data_device)
        xb = self.X[e[:, None] + self.offsets]
        yb = self.rul[e] / P.RUL_CAP
        return xb.to(self.device, non_blocking=True).float(), yb.to(self.device, non_blocking=True)
