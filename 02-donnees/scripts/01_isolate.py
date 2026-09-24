"""
Etape 1 - Isolation : reduire le dataset Backblaze brut a UNE seule reference
(un seul modele de disque) et aux colonnes utiles.

C'est l'equivalent, pour Backblaze, de choisir FD001 chez C-MAPSS : on se
donne un perimetre simple et coherent (un seul modele, donc un seul jeu
d'attributs SMART actifs) avant de nettoyer et pretraiter.

Ecriture en streaming (fichier par fichier) pour rester leger en memoire
meme sur un dataset source de plusieurs dizaines de Go : rien n'est garde
en RAM au-dela du fichier en cours de traitement.

Le tri des colonnes SMART peu remplies / quasi constantes n'est PAS fait ici
(ca demanderait d'avoir tout le dataset en memoire) : c'est le role de
scripts/02_clean.py, qui travaille sur le fichier isole (donc bien plus
petit) et peut se permettre de tout charger.

Usage :
    python scripts/01_isolate.py --model ST4000DM000
    python scripts/01_isolate.py            # auto: prend le top 1 de
                                              # reports/00_scan_modeles.csv

Entree  : data/raw/**/*.csv (fichiers Backblaze bruts)
Sortie  : data/isolated/isolated_<model>.csv
          reports/01_isolation_report.md (justification des choix)
"""

import argparse
import glob
import os
import time

import pandas as pd

RAW_DIR = os.path.join("data", "raw")
ISOLATED_DIR = os.path.join("data", "isolated")
SCAN_REPORT = os.path.join("reports", "00_scan_modeles.csv")
SCAN_REPORT_SAMPLE = os.path.join("reports", "00_scan_modeles_echantillon.csv")
OUT_REPORT = os.path.join("reports", "01_isolation_report.md")

BASE_COLS = ["date", "serial_number", "model", "capacity_bytes", "failure"]


def pick_model(explicit_model):
    if explicit_model:
        return explicit_model
    for path in (SCAN_REPORT, SCAN_REPORT_SAMPLE):
        if os.path.exists(path):
            scan = pd.read_csv(path)
            if not scan.empty:
                return scan.iloc[0]["model"]
    raise SystemExit(
        "Aucun --model fourni et aucun rapport de scan trouve. "
        "Lance d'abord scripts/00_scan_models.py, ou precise --model."
    )


def find_raw_files():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True))
    if not files:
        raise SystemExit(f"Aucun fichier .csv trouve dans {RAW_DIR}/.")
    return files


def scan_raw_columns(files):
    """Lit seulement l'en-tete de chaque fichier (rapide) pour batir l'union
    des colonnes smart_*_raw disponibles sur toute la periode."""
    raw_cols = set()
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip().split(",")
        raw_cols.update(c for c in header if c.startswith("smart_") and c.endswith("_raw"))
    return sorted(raw_cols)


def isolate(model):
    files = find_raw_files()
    print(f"Isolation du modele : {model}")
    print(f"{len(files)} fichier(s) source.\n")

    print("Pre-passe : lecture des en-tetes pour batir la liste des colonnes SMART...")
    raw_cols = scan_raw_columns(files)
    final_cols = BASE_COLS + raw_cols
    print(f"{len(raw_cols)} colonnes smart_*_raw distinctes trouvees sur la periode.\n")

    os.makedirs(ISOLATED_DIR, exist_ok=True)
    out_path = os.path.join(ISOLATED_DIR, f"isolated_{model}.csv")

    rows_total = 0
    failures_total = 0
    serials = set()
    failed_serials = set()
    date_min, date_max = None, None
    header_written = False
    t0 = time.time()

    with open(out_path, "w", encoding="utf-8", newline="") as out_f:
        for i, path in enumerate(files, 1):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                header = f.readline().strip().split(",")
            usecols = [c for c in final_cols if c in header]

            df = pd.read_csv(path, usecols=usecols)
            df = df[df["model"] == model]
            if not df.empty:
                df = df.reindex(columns=final_cols)  # colonnes absentes de ce fichier -> NaN
                df.to_csv(out_f, index=False, header=not header_written)
                header_written = True

                rows_total += len(df)
                failures_total += int(df["failure"].sum())
                serials.update(df["serial_number"].unique())
                failed_serials.update(df.loc[df["failure"] == 1, "serial_number"].unique())
                dmin, dmax = df["date"].min(), df["date"].max()
                date_min = dmin if date_min is None else min(date_min, dmin)
                date_max = dmax if date_max is None else max(date_max, dmax)

            if i % 50 == 0 or i == len(files):
                elapsed = time.time() - t0
                print(f"  ... {i}/{len(files)} fichiers traites "
                      f"({rows_total} lignes retenues, {elapsed:.0f}s ecoulees)")

    if rows_total == 0:
        os.remove(out_path)
        raise SystemExit(f"Aucune ligne trouvee pour le modele '{model}'. Verifie le nom exact.")

    os.makedirs("reports", exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(f"# Rapport d'isolation - modele {model}\n\n")
        f.write(f"- Fichiers sources lus : {len(files)}\n")
        f.write(f"- Lignes retenues (modele = {model}) : {rows_total}\n")
        f.write(f"- Disques uniques : {len(serials)}\n")
        f.write(f"- Lignes en panne (failure=1) : {failures_total}\n")
        f.write(f"- Disques ayant connu une panne : {len(failed_serials)}\n")
        f.write(f"- Periode couverte : {date_min} -> {date_max}\n\n")
        f.write(f"## Colonnes SMART incluses ({len(raw_cols)})\n")
        f.write(
            "Toutes les colonnes smart_*_raw vues sur la periode sont conservees ici "
            "(NaN quand une annee ne rapportait pas cet attribut). Le tri des colonnes "
            "peu remplies / constantes se fait dans l'etape suivante (02_clean.py), "
            "sur ce fichier deja isole donc bien plus petit.\n\n"
        )
        for c in raw_cols:
            f.write(f"- {c}\n")

    print(f"\nFichier isole ecrit : {out_path}")
    print(f"Rapport ecrit       : {OUT_REPORT}")
    print(
        f"\nResume : {rows_total} lignes, {len(serials)} disques, {failures_total} pannes "
        f"({len(failed_serials)} disques distincts en panne), {len(raw_cols)} colonnes SMART."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Nom exact du modele Backblaze a isoler")
    args = parser.parse_args()
    isolate(pick_model(args.model))
