# 3. Modélisation (L2)

Quatre modèles prédisent la RUL (jours restants avant la panne, plafonnée à 125)
à partir de fenêtres de 30 jours des 24 attributs SMART, puis lèvent une alerte
quand la panne est jugée imminente (RUL <= 30 jours).

| Modèle | Dossier | Responsable | Statut |
|---|---|---|---|
| XGBoost | `xgboost/` | | à ajouter |
| CNN-1D | [`cnn1d/`](cnn1d/) | Chaimae | terminé (v2) |
| LSTM | `lstm/` | | à ajouter |
| Transformer | `transformer/` | | à ajouter |

## Code commun : `commun/`

Les 4 modèles suivent le même protocole de comparaison. Le code qui l'applique
est partagé pour que les résultats soient comparables :

| Fichier | Rôle |
|---|---|
| `protocole.py` | Règles communes : fenêtre 30 j, RUL 125, horizon 30 j, trous de 7 j tolérés, ratio 1 dégradation pour 2 saines, graine 42 |
| `donnees.py` | Téléchargement Hugging Face, chargement d'un split en float16, corrections (lignes après la panne, trous de dates) |
| `fenetres.py` | Fenêtres construites à la volée sur GPU (`window_ends`, `sample_ends`, `WindowStore`) |
| `evaluation.py` | Critère de sélection, choix du seuil sur la validation, métriques du test, format des fichiers de prédictions |

Un nouveau modèle PyTorch (LSTM, Transformer) n'a qu'à fournir sa classe : le
reste s'importe depuis `commun/`, comme dans `cnn1d/train.py`.

## Fichiers volumineux

Les poids des modèles et les fichiers de prédictions ne sont pas dans Git. Ils
sont sur Hugging Face (voir le README de chaque modèle).
