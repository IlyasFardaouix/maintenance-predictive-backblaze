# Backblaze - Prétraitement (Livrable L1)

Pipeline en 3 étapes pour isoler puis nettoyer un sous-ensemble exploitable
du dataset Backblaze, avant l'étape de prétraitement final (RUL, normalisation,
fenêtrage) faite séparément pour le LSTM.

## Structure

```
data cleaning/
├── data/
│   ├── raw/        <- dépose ici les fichiers Backblaze bruts (.csv)
│   ├── isolated/   <- sortie étape 1 : un seul modèle de disque
│   └── clean/      <- sortie étape 2 : dataset nettoyé
├── reports/        <- rapports générés (justification des choix, pour le rapport)
├── scripts/
│   ├── 00_scan_models.py
│   ├── 01_isolate.py
│   └── 02_clean.py
├── .venv/          <- environnement Python dédié à ce sous-projet
└── requirements.txt
```

## 1. Déposer les données

Place les fichiers Backblaze bruts (les `.csv` du zip trimestriel officiel,
ou l'export Kaggle) dans `data/raw/`. Le script accepte :
- plusieurs fichiers (un par jour ou un par trimestre), dans des sous-dossiers ou non ;
- un seul gros fichier déjà fusionné.

Colonnes attendues (format standard Backblaze) : `date`, `serial_number`,
`model`, `capacity_bytes`, `failure`, `smart_X_raw`, `smart_X_normalized`.

## 2. Lancer le pipeline

Depuis ce dossier (`data cleaning/`) :

```bash
# Etape 0 - repérer quel modèle choisir comme "référence unique"
.venv/Scripts/python.exe scripts/00_scan_models.py

# Etape 1 - isoler ce modèle + les colonnes SMART pertinentes
.venv/Scripts/python.exe scripts/01_isolate.py --model ST4000DM000

# Etape 2 - nettoyer le dataset isolé
.venv/Scripts/python.exe scripts/02_clean.py --model ST4000DM000
```

(remplace `ST4000DM000` par le modèle réellement choisi, donné par l'étape 0)

## 3. Résultats

- `data/isolated/isolated_<model>.csv` : un seul modèle, colonnes utiles seulement
- `data/clean/clean_<model>.csv` : doublons retirés, colonnes constantes retirées,
  trous comblés par disque, disques trop courts écartés
- `reports/*.md` : le détail chiffré de chaque décision, réutilisable directement
  dans la section "Données et prétraitement (L1)" du rapport final

## Prochaine étape (pas encore faite ici)

Le vrai "prétraitement" (calcul de la RUL, plafonnement, normalisation
train-only, fenêtres glissantes) est volontairement **séparé** du cleaning :
il sera fait dans un script `03_preprocess.py` une fois le cleaning validé,
pour ne pas mélanger "rendre les données fiables" et "les préparer pour le modèle".
