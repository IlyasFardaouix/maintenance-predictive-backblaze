# 3. Modélisation

Responsables : **Tom** (baselines, modèle complémentaire GRU/CNN) et **Karim** (LSTM, optimisation)

## Ce qui doit arriver ici (Livrable L2, échéance fin S3)

- Baseline (régression linéaire ou Random Forest) entraînée sur les sorties de `02-donnees/data/clean/` (puis du futur `03_preprocess.py`)
- Modèle LSTM (squelette Keras indicatif : cahier des charges page 18)
- Éventuellement un second modèle (GRU ou CNN 1D)
- Fichier(s) modèle sauvegardé(s) (`.keras`)
- Notebook ou script d'entraînement, hyperparamètres documentés

Entrée attendue : le dataset nettoyé + prétraité (RUL calculée, normalisé, fenêtré) produit à partir de `02-donnees/`.
