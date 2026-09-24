"""
Etape 0 - Scan des modeles disponibles dans le dataset Backblaze brut.

But : repondre a la question "quelle est ma seule reference (modele de disque) ?"
en comptant, pour chaque `model`, le nombre de lignes, de disques uniques et de
pannes (failure=1) disponibles. C'est l'equivalent, pour Backblaze, du choix de
FD001 dans le cahier des charges (une seule "condition" simple et coherente).

Usage :
    .venv/Scripts/python.exe scripts/00_scan_models.py

Entree  : tous les fichiers .csv trouves (recursivement) dans data/raw/
Sortie  : reports/00_scan_modeles.csv (classement des modeles)
          + affichage du top 15 dans la console
"""

import argparse
import glob
import os
from collections import defaultdict

import pandas as pd

RAW_DIR = os.path.join("data", "raw")
REPORT_PATH = os.path.join("reports", "00_scan_modeles.csv")

USECOLS = ["date", "serial_number", "model", "failure"]


def find_raw_files(sample_per_month=None):
    files = sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True))
    if not files:
        raise SystemExit(
            f"Aucun fichier .csv trouve dans {RAW_DIR}/. "
            "Depose les fichiers Backblaze (ou le contenu du zip trimestriel) dedans."
        )
    if sample_per_month:
        by_month = defaultdict(list)
        for f in files:
            # nom de fichier attendu : YYYY-MM-DD.csv
            key = os.path.basename(f)[:7]  # "YYYY-MM"
            by_month[key].append(f)
        sampled = []
        for key in sorted(by_month):
            month_files = sorted(by_month[key])
            step = max(1, len(month_files) // sample_per_month)
            sampled.extend(month_files[::step][:sample_per_month])
        files = sorted(sampled)
    return files


def scan(sample_per_month=None):
    files = find_raw_files(sample_per_month)
    label = f" (echantillon : {sample_per_month}/mois)" if sample_per_month else " (scan complet)"
    print(f"{len(files)} fichier(s) source a traiter{label}\n")

    per_model = {}  # model -> dict(rows, failures, serials:set, date_min, date_max)

    for i, path in enumerate(files, 1):
        try:
            df = pd.read_csv(path, usecols=USECOLS, engine="pyarrow")
        except ValueError:
            # fichier sans une des colonnes attendues -> on le signale et on saute
            print(f"  [!] {os.path.basename(path)} : colonnes attendues absentes, ignore")
            continue

        for model, g in df.groupby("model"):
            entry = per_model.setdefault(
                model,
                {"rows": 0, "failures": 0, "serials": set(), "date_min": None, "date_max": None},
            )
            entry["rows"] += len(g)
            entry["failures"] += int(g["failure"].sum())
            entry["serials"].update(g["serial_number"].unique())
            dmin, dmax = g["date"].min(), g["date"].max()
            entry["date_min"] = dmin if entry["date_min"] is None else min(entry["date_min"], dmin)
            entry["date_max"] = dmax if entry["date_max"] is None else max(entry["date_max"], dmax)

        if i % 10 == 0 or i == len(files):
            print(f"  ... {i}/{len(files)} fichiers traites")

    rows = []
    for model, e in per_model.items():
        rows.append(
            {
                "model": model,
                "nb_lignes": e["rows"],
                "nb_disques_uniques": len(e["serials"]),
                "nb_pannes": e["failures"],
                "date_min": e["date_min"],
                "date_max": e["date_max"],
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["nb_pannes", "nb_disques_uniques"], ascending=False
    )

    out_path = REPORT_PATH.replace(".csv", "_echantillon.csv") if sample_per_month else REPORT_PATH
    os.makedirs("reports", exist_ok=True)
    summary.to_csv(out_path, index=False)

    print(f"\nClassement ecrit dans {out_path}\n")
    print("Top 15 modeles (tries par nombre de pannes observees) :\n")
    print(summary.head(15).to_string(index=False))

    if len(summary) > 0:
        best = summary.iloc[0]
        print(
            f"\nCandidat recommande : {best['model']} "
            f"({int(best['nb_pannes'])} pannes, {int(best['nb_disques_uniques'])} disques, "
            f"{int(best['nb_lignes'])} lignes)."
        )
        print("-> a confirmer/justifier dans le rapport (section 'sélection des capteurs / choix du modèle').")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-per-month",
        type=int,
        default=None,
        help="Ne lire que N fichiers par mois (echantillonnage rapide) au lieu de tout",
    )
    args = parser.parse_args()
    scan(args.sample_per_month)
