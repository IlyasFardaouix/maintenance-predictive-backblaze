# Journal d'expériences - CNN-1D

Choix de la version : sur le **critère de validation** (moyenne du RMSE en
dégradation et du RMSE sur les sains), jamais sur le test.

| Essai | Date | Changement | Meilleure epoch / arrêt | Critère val (j) | F1 alerte val | Décision |
|---|---|---|---|---|---|---|
| v2 | 08/11/2026 | Protocole commun (lignes après panne supprimées, trous de 7 j tolérés, critère équilibré). LR 1e-3, patience 5 | 2 / 7 | **33,26** | **0,164** | **Retenu** |
| v2-lr | 08/11/2026 | LR 3e-4, patience 10, 40 epochs max | 2 / 12 | 34,06 | 0,113 | Non retenu |

Observation : dans les deux essais, la validation oscille d'une epoch à l'autre
entre « bon sur les disques sains » et « bon sur les dégradations ». Un taux
d'apprentissage plus faible ne corrige pas ce comportement : l'instabilité vient
de la faiblesse du signal de dégradation, pas du réglage de l'optimisation.
