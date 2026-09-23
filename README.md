# Projet 2 — Maintenance prédictive à partir de séries temporelles capteurs

Projet d'équipe (Industrie 4.0, 6 personnes, 4 semaines) : prédire la durée de
vie restante (RUL) à partir de données de capteurs, avec un modèle de
référence (baseline) et un modèle LSTM.

Cahier des charges complet : [`01-gestion-projet/Projet2_Maintenance_predictive_plan_4_semaines_6_membres.pdf`](01-gestion-projet/Projet2_Maintenance_predictive_plan_4_semaines_6_membres.pdf)

**Jeu de données :** [Backblaze Hard Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)
(choisi à la place du NASA C-MAPSS FD001 suggéré par le sujet — à justifier
dans la section 1 du rapport final).

## Équipe et répartition (voir RACI, cahier des charges p.12)

| Dossier | Lot (WBS) | Responsable(s) | Livrable |
|---|---|---|---|
| [`01-gestion-projet/`](01-gestion-projet/) | Gestion de projet | Alex | Cahier des charges, planning, suivi |
| [`02-donnees/`](02-donnees/) | Données et prétraitement | **Sarah** | L1 |
| [`03-modelisation/`](03-modelisation/) | Modélisation | Karim (LSTM), Tom (baselines) | L2 |
| [`04-evaluation/`](04-evaluation/) | Évaluation | Léa | L3 |
| [`05-interpretation-metier/`](05-interpretation-metier/) | Interprétation métier | Inès | L4 |
| [`06-livrables-finaux/`](06-livrables-finaux/) | Rapport, slides, démo | Toute l'équipe | L5, L7 |

Chaque dossier a son propre README avec le détail de ce qui doit y arriver.

## Enchaînement du pipeline

```
02-donnees/  →  03-modelisation/  →  04-evaluation/  →  05-interpretation-metier/  →  06-livrables-finaux/
(clean data)    (baseline + LSTM)    (métriques)         (seuil d'alerte, coûts)       (rapport, démo)
```

## État d'avancement

- ✅ `02-donnees/` : pipeline d'isolation (un seul modèle de disque) et de nettoyage prêt
  (voir [`02-donnees/README.md`](02-donnees/README.md)). En attente du dépôt du dataset brut.
- ⬜ Les autres dossiers sont des emplacements réservés, à remplir par chaque binôme.

## Outils

Chaque dossier peut avoir son propre environnement Python (ex. `02-donnees/.venv/`,
`02-donnees/requirements.txt`) puisque les besoins diffèrent (pandas/numpy pour
le nettoyage, TensorFlow/Keras pour la modélisation). Rien de tout ça n'est
versionné (voir `.gitignore` à la racine) : chacun recrée son environnement
localement avec le `requirements.txt` de son dossier.
