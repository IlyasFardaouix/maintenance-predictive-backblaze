# Projet 2 — Maintenance prédictive à partir de séries temporelles capteurs

Projet d'équipe (Industrie 4.0, 6 personnes, 4 semaines) : prédire la durée de
vie restante (RUL) à partir de données de capteurs, avec un modèle de
référence (baseline) et un modèle LSTM.

Cahier des charges complet : [`01-gestion-projet/Projet2_Maintenance_predictive_plan_4_semaines_6_membres.pdf`](01-gestion-projet/Projet2_Maintenance_predictive_plan_4_semaines_6_membres.pdf)

**Jeu de données :** [Backblaze Hard Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)
(choisi à la place du NASA C-MAPSS FD001 suggéré par le sujet — à justifier
dans la section 1 du rapport final).

## Organisation Git de l'équipe

Convention du cahier des charges (p.15) : **une branche par personne, fusion
après relecture** — jamais de travail direct sur `main`. Chaque dossier de lot
(voir WBS, cahier des charges p.12) est ajouté au dépôt par son binôme
responsable, via sa propre branche, quand son travail est prêt :

| Dossier | Lot (WBS) | Responsable(s) | Livrable | Statut |
|---|---|---|---|---|
| [`01-gestion-projet/`](01-gestion-projet/) | Gestion de projet | Alex | — | ✅ présent |
| [`02-donnees/`](02-donnees/) | Données et prétraitement | **Sarah** | L1 | ✅ présent |
| `03-modelisation/` | Modélisation | Karim (LSTM), Tom (baselines) | L2 | ⬜ à venir (branche à part) |
| `04-evaluation/` | Évaluation | Léa | L3 | ⬜ à venir (branche à part) |
| `05-interpretation-metier/` | Interprétation métier | Inès | L4 | ⬜ à venir (branche à part) |
| `06-livrables-finaux/` | Rapport, slides, démo | Toute l'équipe | L5, L7 | ⬜ à venir (branche à part) |

## État d'avancement

- ✅ `02-donnees/` : pipeline d'isolation (un seul modèle de disque) et de nettoyage prêt
  (voir [`02-donnees/README.md`](02-donnees/README.md)). En attente du dépôt du dataset brut.
- Les autres lots seront ajoutés par leur responsable via une branche dédiée, puis
  fusionnés dans `main` après relecture croisée.

## Outils

Chaque dossier peut avoir son propre environnement Python (ex. `02-donnees/.venv/`,
`02-donnees/requirements.txt`) puisque les besoins diffèrent (pandas/numpy pour
le nettoyage, TensorFlow/Keras pour la modélisation). Rien de tout ça n'est
versionné (voir `.gitignore` à la racine) : chacun recrée son environnement
localement avec le `requirements.txt` de son dossier.
