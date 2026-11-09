"""Règles communes à tous les modèles (fiche protocole commun).

Toute modification d'une de ces valeurs doit être décidée en groupe :
elle change les résultats des 4 modèles.
"""

# Données
HF_DATASET_REPO = "IlyasFardaouixx/backblaze-st4000dm000-predictive-maintenance"
HF_DATASET_FILE = "preprocessed_ST4000DM000.parquet"

# Problème
WINDOW = 30          # longueur de la fenêtre glissante (jours)
RUL_CAP = 125        # plafond de RUL : au-delà, le disque est considéré sain
ALERT_HORIZON = 30   # panne imminente = RUL réelle <= 30 jours
MAX_GAP_DAYS = 7     # écart de dates toléré dans une même série

# Déséquilibre : nombre de fenêtres saines tirées pour 1 fenêtre en dégradation
HEALTHY_RATIO = 2

# Reproductibilité
SEED = 42
