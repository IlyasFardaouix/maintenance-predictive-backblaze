"""Hyperparamètres du CNN-1D, version 2 (version finale retenue)."""
from types import SimpleNamespace

CFG = SimpleNamespace(
    # Architecture
    CHANNELS=(64, 64, 128),
    KERNEL=3,
    DROPOUT=0.3,
    # Entraînement
    BATCH_SIZE=1024,
    EPOCHS=30,
    LR=1e-3,
    WEIGHT_DECAY=1e-4,
    PATIENCE=5,
    MAX_TRAIN_MINUTES=150,     # arrêt propre avant d'épuiser le quota GPU Kaggle
    HUBER_DELTA=0.1,           # sur la cible /125, soit environ 12 jours
    # Mémoire et évaluation
    STORE_DTYPE="float16",     # données min-max dans [0, 1] : float16 suffit
    EVAL_BATCH=16384,
)
