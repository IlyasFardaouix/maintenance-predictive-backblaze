# Journal des expériences — XGBoost

## v1 — Baseline (single-pass, disques en panne uniquement)
- Entraînement sur les seuls disques en panne du train
- RMSE dégradés : ~68 j · Détection : ~45%
- Problème : sous-représentation des disques sains → trop de fausses alarmes

## v2 — Équilibrage 1×2 (single-pass)
- Ajout d'un réservoir de disques sains (ratio 2:1)
- Échantillonnage limité à un sous-ensemble de disques sains
- RMSE dégradés : ~55 j · Détection : ~58%

## v3 — Retenue  (2 passes, tous les 25 223 disques train)
- Passe 1 : tous les disques en panne → 329 200 fenêtres dégradées
- Passe 2 : reservoir sampling sur 22 454 disques sains → 658 400 fenêtres saines
- **RMSE dégradés : 49.8 j · Détection : 62.5% · Avance : 55 j · FA : 4.1%**
- Meilleure itération : 66 (early stopping sur balanced RMSE val)