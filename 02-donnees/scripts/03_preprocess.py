"""
Etape 3 - Pretraitement : calcul de la RUL, split train/val/test par disque,
normalisation min-max (parametres appris sur le train uniquement).

Volontairement separe du nettoyage (02_clean.py), qui rendait les donnees
fiables sans les transformer pour un modele.

Fenetrage : PAS fait ici. Materialiser toutes les fenetres glissantes a
l'avance exploserait en taille (des dizaines de millions de fenetres de 30
jours x 24 capteurs = plusieurs dizaines de milliards de valeurs). La
fonction make_windows() du squelette LSTM (cahier des charges, C.3) est
prevue pour construire les fenetres a la volee, en memoire, au moment de
l'entrainement (03-modelisation/), a partir du fichier produit ici.

RUL (equivalent Backblaze de la RUL C-MAPSS, cf. echange precedent) :
  - disque qui finit par tomber en panne : RUL = jours jusqu'a la panne,
    plafonne a RUL_CAP (meme logique que C-MAPSS, cahier des charges p.17)
  - disque jamais observe en panne (censure) : RUL = RUL_CAP partout, en
    l'absence d'information contraire. A documenter comme limite (donnee
    censuree, pas une vraie mesure) dans l'interpretation metier (L4).

Deux passes en streaming (memoire bornee, meme principe que 02_clean.py) :
  Passe A : RUL + split (train/val/test par disque, hash stable) + calcul
            des min/max par colonne SMART sur le train uniquement.
  Passe B : normalisation min-max avec les parametres de la passe A.

Usage :
    python scripts/03_preprocess.py --model ST4000DM000

Entree  : data/clean/clean_<model>.csv
Sortie  : data/preprocessed/preprocessed_<model>.csv
          data/preprocessed/normalization_params_<model>.json
          reports/03_preprocessing_report.md
"""

import argparse
import json
import os
import zlib

import pandas as pd

CLEAN_DIR = os.path.join("data", "clean")
PREPROC_DIR = os.path.join("data", "preprocessed")
OUT_REPORT = os.path.join("reports", "03_preprocessing_report.md")

CHUNKSIZE = 1_000_000
RUL_CAP = 125  # meme valeur que celle discutee pour C-MAPSS (cahier des charges p.17)
TRAIN_PCT, VAL_PCT = 70, 15  # test = reste (15%)


def split_of(serial):
    h = zlib.crc32(serial.encode("utf-8")) % 100
    if h < TRAIN_PCT:
        return "train"
    if h < TRAIN_PCT + VAL_PCT:
        return "val"
    return "test"


def add_rul_and_split(df, split_cache):
    df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)

    fail_date_col = df["date"].where(df["failure"] == 1)
    fail_date_per_group = df.assign(_fd=fail_date_col).groupby("serial_number", sort=False)["_fd"].transform("min")

    rul = (fail_date_per_group - df["date"]).dt.days
    rul = rul.fillna(RUL_CAP).clip(lower=0, upper=RUL_CAP)
    df["rul"] = rul.astype(int)

    new_serials = [s for s in df["serial_number"].unique() if s not in split_cache]
    for s in new_serials:
        split_cache[s] = split_of(s)
    df["split"] = df["serial_number"].map(split_cache)

    return df


def _process_and_write(df, out_f, header_written_flag, split_cache, col_min, col_max,
                        smart_cols, split_disk_counts, split_row_counts, stats):
    df = add_rul_and_split(df, split_cache)

    stats["rows_total"] += len(df)
    stats["rul_at_cap"] += int((df["rul"] >= RUL_CAP).sum())
    for sp in ("train", "val", "test"):
        m = df["split"] == sp
        split_row_counts[sp] += int(m.sum())
        split_disk_counts[sp].update(df.loc[m, "serial_number"].unique())

    train_df = df[df["split"] == "train"]
    if not train_df.empty:
        for c in smart_cols:
            cmin, cmax = train_df[c].min(), train_df[c].max()
            col_min[c] = cmin if col_min[c] is None else min(col_min[c], cmin)
            col_max[c] = cmax if col_max[c] is None else max(col_max[c], cmax)

    df.to_csv(out_f, index=False, header=not header_written_flag[0])
    header_written_flag[0] = True


def run_pass_a(in_path, smart_cols):
    tmp_path = os.path.join(PREPROC_DIR, "_tmp_with_rul.csv")
    os.makedirs(PREPROC_DIR, exist_ok=True)

    split_cache = {}
    col_min = {c: None for c in smart_cols}
    col_max = {c: None for c in smart_cols}
    split_disk_counts = {"train": set(), "val": set(), "test": set()}
    split_row_counts = {"train": 0, "val": 0, "test": 0}
    stats = {"rows_total": 0, "rul_at_cap": 0}
    header_written_flag = [False]

    reader = pd.read_csv(in_path, chunksize=CHUNKSIZE, parse_dates=["date"])

    pending = None
    with open(tmp_path, "w", encoding="utf-8", newline="") as out_f:
        prev_chunk = None
        for chunk in reader:
            if prev_chunk is not None:
                combined = pd.concat([pending, prev_chunk], ignore_index=True) if pending is not None else prev_chunk
                boundary_serial = chunk["serial_number"].iloc[0]
                is_boundary = combined["serial_number"] == combined["serial_number"].iloc[-1]
                # ne garde en attente que si le dernier disque du batch pourrait
                # continuer dans le chunk suivant (meme serial_number en tete du suivant)
                if combined["serial_number"].iloc[-1] == boundary_serial:
                    pending = combined[is_boundary]
                    complete = combined[~is_boundary]
                else:
                    pending = None
                    complete = combined
                if not complete.empty:
                    _process_and_write(complete, out_f, header_written_flag, split_cache,
                                        col_min, col_max, smart_cols, split_disk_counts,
                                        split_row_counts, stats)
                print(f"  ... {stats['rows_total']} lignes (RUL + split) ecrites")
            prev_chunk = chunk

        # dernier chunk : plus rien apres, tout est complet
        if prev_chunk is not None:
            combined = pd.concat([pending, prev_chunk], ignore_index=True) if pending is not None else prev_chunk
            _process_and_write(combined, out_f, header_written_flag, split_cache,
                                col_min, col_max, smart_cols, split_disk_counts,
                                split_row_counts, stats)
            print(f"  ... {stats['rows_total']} lignes (RUL + split) ecrites (dernier lot)")

    return tmp_path, col_min, col_max, split_disk_counts, split_row_counts, stats


def run_pass_b(tmp_path, out_path, smart_cols, col_min, col_max):
    header_written = False
    with open(out_path, "w", encoding="utf-8", newline="") as out_f:
        for chunk in pd.read_csv(tmp_path, chunksize=CHUNKSIZE):
            for c in smart_cols:
                lo, hi = col_min[c], col_max[c]
                if hi is None or lo is None or hi == lo:
                    chunk[c] = 0.0
                else:
                    chunk[c] = ((chunk[c] - lo) / (hi - lo)).clip(0.0, 1.0)
            chunk.to_csv(out_f, index=False, header=not header_written)
            header_written = True


def preprocess(model):
    in_path = os.path.join(CLEAN_DIR, f"clean_{model}.csv")
    if not os.path.exists(in_path):
        raise SystemExit(f"{in_path} introuvable. Lance d'abord scripts/02_clean.py.")

    header = pd.read_csv(in_path, nrows=0).columns.tolist()
    smart_cols = [c for c in header if c.startswith("smart_")]

    print("Passe A : calcul de la RUL, split train/val/test, min/max (train) par colonne...")
    tmp_path, col_min, col_max, split_disk_counts, split_row_counts, stats = run_pass_a(in_path, smart_cols)

    print("\nPasse B : normalisation min-max...")
    os.makedirs(PREPROC_DIR, exist_ok=True)
    out_path = os.path.join(PREPROC_DIR, f"preprocessed_{model}.csv")
    run_pass_b(tmp_path, out_path, smart_cols, col_min, col_max)
    os.remove(tmp_path)

    params_path = os.path.join(PREPROC_DIR, f"normalization_params_{model}.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "min-max",
                "fitted_on": "train split only",
                "rul_cap": RUL_CAP,
                "columns": {c: {"min": col_min[c], "max": col_max[c]} for c in smart_cols},
            },
            f,
            indent=2,
        )

    os.makedirs("reports", exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(f"# Rapport de pretraitement - modele {model}\n\n")
        f.write(f"- Lignes traitees : {stats['rows_total']}\n")
        f.write(f"- Plafond de RUL : {RUL_CAP} jours\n")
        f.write(
            f"- Lignes au plafond (RUL={RUL_CAP}, disque loin d'une panne ou jamais "
            f"observe en panne = donnee censuree) : {stats['rul_at_cap']} "
            f"({stats['rul_at_cap'] / stats['rows_total'] * 100:.1f}%)\n\n"
        )
        f.write("## Split train / validation / test (par disque, hash stable du serial_number)\n\n")
        f.write("| Split | Disques | Lignes |\n|---|---|---|\n")
        for sp in ("train", "val", "test"):
            f.write(f"| {sp} | {len(split_disk_counts[sp])} | {split_row_counts[sp]} |\n")
        f.write(
            "\nSplit fait par disque entier (jamais ligne par ligne) pour eviter toute "
            "fuite de donnees : aucun disque n'apparait dans deux splits differents.\n"
        )
        f.write(
            "\n## Normalisation\nMin-max par colonne SMART, parametres (min/max) calcules "
            "**uniquement sur le split train**, puis appliques tels quels a val et test "
            "(clip a [0,1] si une valeur de val/test depasse la plage vue en train). "
            f"Parametres sauvegardes dans `{os.path.basename(params_path)}`.\n"
        )
        f.write(
            "\n## Limite : RUL censuree\nPour un disque jamais observe en panne dans la "
            "periode disponible (2016-2022), la vraie RUL est inconnue : on ne sait pas "
            "s'il tombera en panne demain ou dans 10 ans apres la fin de la periode "
            "observee. Le choix ici est de lui assigner RUL = plafond partout (traite "
            "comme 'loin d'une panne'), une simplification a mentionner explicitement "
            "dans les limites du modele (L4).\n"
        )
        f.write(
            "\n## Fenetrage : non materialise ici\nGenerer a l'avance toutes les fenetres "
            "glissantes possibles (30 jours, stride 1) sur ce volume produirait un fichier "
            "d'une taille ingerable (dizaines de milliards de valeurs). La fonction "
            "`make_windows()` du squelette LSTM (cahier des charges, C.3) doit etre "
            "appelee au moment de l'entrainement, a partir de ce fichier, pour construire "
            "les fenetres en memoire (par lot ou par disque).\n"
        )

    print(f"\nFichier pretraite ecrit : {out_path}")
    print(f"Parametres de normalisation : {params_path}")
    print(f"Rapport ecrit : {OUT_REPORT}")
    print(f"\nResume : {stats['rows_total']} lignes | train={split_row_counts['train']} "
          f"val={split_row_counts['val']} test={split_row_counts['test']} | "
          f"{stats['rul_at_cap']} lignes au plafond de RUL")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    preprocess(args.model)
