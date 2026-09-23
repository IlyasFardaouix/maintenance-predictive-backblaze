# Maintenance prédictive - séries temporelles capteurs

Prédire la durée de vie restante (RUL) de disques durs à partir de leurs
données SMART, avec plusieurs modèles, dont un LSTM.

**Jeu de données :** [Backblaze Hard Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)

## Structure

- [`01-gestion-projet/`](01-gestion-projet/) : cahier des charges
- [`02-donnees/`](02-donnees/) : isolation et nettoyage des données ([détails](02-donnees/README.md))

Les autres dossiers (modélisation, évaluation, interprétation métier, livrables)
seront ajoutés par leurs responsables via leur propre branche, puis fusionnés
après relecture.

## Environnement

Chaque dossier a son propre environnement Python (non versionné), voir le
`requirements.txt` correspondant.
