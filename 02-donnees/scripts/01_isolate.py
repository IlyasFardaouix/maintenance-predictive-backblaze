"""
Etape 1 - Isolation : reduire le dataset Backblaze brut a UNE seule reference
(un seul modele de disque) et aux colonnes utiles.

C'est l'equivalent, pour Backblaze, de choisir FD001 chez C-MAPSS : on se
donne un perimetre simple et coherent (un seul modele, donc un seul jeu
d'attributs SMART actifs) avant de nettoyer et pretraiter.

Usage :
    .venv/Scripts/python.exe scripts/01_isolate.py --model ST4000DM000
    .venv/Scripts/python.exe scripts/01_isolate.py            # auto: prend le
                                                                # top 1 de
                                                                # reports/00_scan_modeles.csv

Entree  : data/raw/**/*.csv (fichiers Backblaze bruts)
Sortie  : data/isolated/isolated_<model>.csv
          reports/01_isolation_report.md (justification des choix)
"""

import argparse
import glob
import os

import pandas as pd

RAW_DIR = os.path.join("data", "raw")
ISOLATED_DIR = os.path.join("data", "isolated")
SCAN_REPORT = os.path.join("reports", "00_scan_modeles.csv")
OUT_REPORT = os.path.join("reports", "01_isolation_report.md")

BASE_COLS = ["date", "serial_number", "model", "capacity_bytes", "failure"]
MAX_MISSING_RATE = 0.90  # une colonne smart_*_raw gardee seulement si < 90% de NaN


def pick_model(explicit_model):
    if explicit_model:
        return explicit_model
    if not os.path.exists(SCAN_REPORT):
        raise SystemExit(
            "Aucun --model fourni et reports/00_scan_modeles.csv introuvable. "
            "Lance d'abord scripts/00_scan_models.py, ou precise --model."
        )
    scan = pd.read_csv(SCAN_REPORT)
    if scan.empty:
        raise SystemExit("reports/00_scan_modeles.csv est vide.")
    return scan.iloc[0]["model"]


def find_raw_files():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True))
    if not files:
        raise SystemExit(f"Aucun fichier .csv trouve dans {RAW_DIR}/.")
    return files


def isolate(model):
    files = find_raw_files()
    print(f"Isolation du modele : {model}")
    print(f"{len(files)} fichier(s) source a lire...\n")

    chunks = []
    rows_total = 0
    for i, path in enumerate(files, 1):
        df = pd.read_csv(path, low_memory=False)
        df = df[df["model"] == model]
        if not df.empty:
            chunks.append(df)
            rows_total += len(df)
        if i % 10 == 0 or i == len(files):
            print(f"  ... {i}/{len(files)} fichiers lus, {rows_total} lignes retenues jusqu'ici")

    if not chunks:
        raise SystemExit(f"Aucune ligne trouvee pour le modele '{model}'. Verifie le nom exact.")

    data = pd.concat(chunks, ignore_index=True)
    del chunks

    # colonnes smart_*_raw disponibles pour ce modele (on ignore les _normalized,
    # redondantes avec les _raw pour la plupart des attributs)
    smart_raw_cols = [c for c in data.columns if c.startswith("smart_") and c.endswith("_raw")]

    missing_rate = data[smart_raw_cols].isna().mean()
    kept_smart = missing_rate[missing_rate < MAX_MISSING_RATE].index.tolist()
    dropped_smart = missing_rate[missing_rate >= MAX_MISSING_RATE].index.tolist()

    final_cols = BASE_COLS + sorted(kept_smart)
    isolated = data[final_cols].copy()

    os.makedirs(ISOLATED_DIR, exist_ok=True)
    out_path = os.path.join(ISOLATED_DIR, f"isolated_{model}.csv")
    isolated.to_csv(out_path, index=False)

    nb_disques = isolated["serial_number"].nunique()
    nb_pannes = int(isolated["failure"].sum())
    nb_disques_en_panne = isolated.loc[isolated["failure"] == 1, "serial_number"].nunique()

    os.makedirs("reports", exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(f"# Rapport d'isolation - modele {model}\n\n")
        f.write(f"- Fichiers sources lus : {len(files)}\n")
        f.write(f"- Lignes retenues (modele = {model}) : {len(isolated)}\n")
        f.write(f"- Disques uniques : {nb_disques}\n")
        f.write(f"- Lignes en panne (failure=1) : {nb_pannes}\n")
        f.write(f"- Disques ayant connu une panne : {nb_disques_en_panne}\n")
        f.write(f"- Periode couverte : {isolated['date'].min()} -> {isolated['date'].max()}\n\n")
        f.write(f"## Colonnes SMART conservees ({len(kept_smart)})\n")
        f.write("Seuil : moins de {:.0f}% de valeurs manquantes pour ce modele.\n\n".format(
            MAX_MISSING_RATE * 100
        ))
        for c in sorted(kept_smart):
            f.write(f"- {c} ({missing_rate[c] * 100:.1f}% manquant)\n")
        f.write(f"\n## Colonnes SMART ecartees ({len(dropped_smart)})\n")
        f.write("Trop de valeurs manquantes pour ce modele (capteur non utilise / non reporte).\n\n")
        for c in sorted(dropped_smart):
            f.write(f"- {c} ({missing_rate[c] * 100:.1f}% manquant)\n")

    print(f"\nFichier isole ecrit : {out_path}")
    print(f"Rapport ecrit       : {OUT_REPORT}")
    print(f"\nResume : {len(isolated)} lignes, {nb_disques} disques, {nb_pannes} pannes, "
          f"{len(kept_smart)} colonnes SMART gardees / {len(smart_raw_cols)} disponibles.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Nom exact du modele Backblaze a isoler")
    args = parser.parse_args()
    isolate(pick_model(args.model))
