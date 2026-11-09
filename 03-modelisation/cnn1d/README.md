# CNN-1D (L2, modèle complémentaire)

Responsable : **Chaimae**. Version finale : **v2**, entraînée sur Kaggle (GPU T4).

## Principe

Le modèle reçoit les 30 derniers jours des 24 attributs SMART d'un disque et
prédit sa RUL. Des filtres de convolution glissent le long des 30 jours et
repèrent des motifs courts de dégradation (par exemple une hausse soudaine des
secteurs réalloués). Une alerte est levée quand la RUL prédite passe sous le
seuil choisi sur la validation.

Architecture : 3 convolutions 1D (64, 64, 128 filtres, noyau 3) avec
normalisation par lot, max pooling, moyenne et maximum globaux, couche dense de
64 neurones, dropout 0,3. Environ 59 000 paramètres.

## Résultats (test complet, 5 419 disques, 8 969 549 fenêtres)

| Indicateur | Valeur |
|---|---|
| RMSE en dégradation (RUL < 125) | 56,8 j (référence « toujours sain » : 73,5 j, soit -23 %) |
| Seuil d'alerte (choisi sur la validation) | 59,8 j |
| F1 de l'alerte / AP | 0,172 / 0,071 |
| Disques défaillants détectés avant la panne | 250 / 572 (43,7 %) |
| Avance moyenne de l'alerte | 41,4 j |
| Fausses alertes sur disques sains | 166 / 4 844 (3,4 %) |
| Meilleure epoch / arrêt | 2 / 7 (early stopping), 0,7 min |

Détails : [`results/metrics_test_cnn1d.json`](results/metrics_test_cnn1d.json),
historique : [`results/history_cnn1d.csv`](results/history_cnn1d.csv),
essais : [`results/journal_experiences.md`](results/journal_experiences.md),
figures : [`results/figures/`](results/figures/).

## Contenu

```
cnn1d/
├── README.md
├── requirements.txt
├── config.py          <- hyperparamètres de la v2
├── model.py           <- classe CNN1D
├── train.py           <- entraînement + évaluation complète (utilise ../commun)
├── notebooks/
│   └── cnn1d_kaggle_v2.ipynb   <- notebook exécuté sur Kaggle, avec ses sorties
└── results/
    ├── metrics_test_cnn1d.json
    ├── config_cnn1d.json
    ├── history_cnn1d.csv
    ├── journal_experiences.md
    └── figures/
```

## Relancer l'entraînement (Kaggle)

1. Nouveau notebook Kaggle, *Settings* : GPU T4, Internet activé.
2. Dans une cellule :

```bash
!git clone https://github.com/IlyasFardaouix/maintenance-predictive-backblaze.git
!pip install -q polars
!python maintenance-predictive-backblaze/03-modelisation/cnn1d/train.py --out /kaggle/working/cnn1d_outputs
```

Le script télécharge le dataset prétraité depuis Hugging Face, applique le
protocole commun, entraîne, évalue sur tout le test et écrit dans `--out` :
poids, configuration, historique, métriques, prédictions val et test, figures.
Pour un fichier local : `--parquet chemin/preprocessed_ST4000DM000.parquet`.

## Fichiers sur Hugging Face

Non versionnés dans Git (trop volumineux) :

| Fichier | Contenu |
|---|---|
| `cnn1d/cnn1d_best.pt` | Poids du modèle retenu (epoch 2) |
| `predictions_val_cnn1d.parquet` | `serial_number`, `date`, `rul_true`, `rul_pred` pour la validation |
| `predictions_test_cnn1d.parquet` | Idem pour le test |
