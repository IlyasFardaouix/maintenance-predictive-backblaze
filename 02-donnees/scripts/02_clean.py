"""
Etape 2 - Nettoyage (cleaning) du dataset isole, en version econome en memoire.

Le fichier isole peut peser plusieurs dizaines de Go (7 ans de donnees pour
un seul modele) : le charger entierement avec pd.read_csv() ferait planter
la machine. On procede donc en 2 passes qui ne gardent jamais tout en RAM :

  Passe 1 (partitionnement) : on relit le fichier isole par blocs
    (chunksize), et pour chaque ligne on la range dans l'un de N_BUCKETS
    fichiers temporaires selon un hash stable du serial_number. Toutes les
    lignes d'un meme disque finissent donc TOUJOURS dans le meme bucket
    (propriete cle : ca permet de traiter chaque disque en entier a la passe
    2, sans jamais couper son historique). Pendant cette meme passe, on
    calcule aussi en ligne (algorithme de Welford) la moyenne/ecart-type de
    chaque colonne SMART, sans stocker les valeurs individuelles.

  Passe 2 (nettoyage par bucket) : chaque bucket est assez petit pour etre
    charge entierement en memoire. Pour chaque bucket : on retire les
    colonnes constantes (ecart-type nul, detectees a la passe 1), on trie
    par disque puis par date, on supprime les doublons exacts, on comble les
    trous PAR DISQUE uniquement (ffill/bfill, jamais entre deux disques -
    piege de fuite de donnees), on ecarte les disques a historique trop
    court, on verifie la coherence de la colonne failure, puis on ecrit le
    resultat dans le fichier propre final (en ajoutant a chaque bucket).

Usage :
    python scripts/02_clean.py --model ST4000DM000

Entree  : data/isolated/isolated_<model>.csv
Sortie  : data/clean/clean_<model>.csv
          reports/02_cleaning_report.md (justification des choix, pour L1)
"""

import argparse
import gc
import math
import os
import shutil
import zlib

import pandas as pd

ISOLATED_DIR = os.path.join("data", "isolated")
CLEAN_DIR = os.path.join("data", "clean")
TMP_DIR = os.path.join("data", "_tmp_buckets")
OUT_REPORT = os.path.join("reports", "02_cleaning_report.md")

N_BUCKETS = 16
CHUNKSIZE = 500_000
MIN_HISTORY = 30  # un disque avec moins de 30 jours de donnees est ecarte
STD_EPSILON = 1e-9  # ecart-type en dessous de ce seuil = colonne consideree constante
MAX_MISSING_RATE = 0.5  # colonne retiree si plus de 50% de valeurs manquantes sur tout le dataset


def bucket_of(serial):
    return zlib.crc32(serial.encode("utf-8")) % N_BUCKETS


class WelfordStats:
    """Moyenne/ecart-type en ligne (Chan et al.), sans stocker les valeurs."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def merge_chunk(self, count, mean, var):
        if count == 0:
            return
        m2_b = var * (count - 1) if count > 1 else 0.0
        if self.n == 0:
            self.n, self.mean, self.m2 = count, mean, m2_b
            return
        delta = mean - self.mean
        n_ab = self.n + count
        self.mean += delta * count / n_ab
        self.m2 += m2_b + delta**2 * self.n * count / n_ab
        self.n = n_ab

    def std(self):
        if self.n < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))


def pass1_partition(in_path, smart_cols):
    # fichiers intermediaires en Parquet (binaire, colonne) plutot qu'en CSV :
    # le formatage texte des flottants par to_csv() est le vrai goulot
    # d'etranglement mesure ici (~10x plus lent que la lecture elle-meme),
    # Parquet l'evite completement.
    # on repart toujours d'un TMP_DIR vide : un ancien lot de buckets (d'une
    # execution precedente, ou d'un run sur un autre fichier) ne doit jamais
    # se meler aux nouveaux, sous peine de corrompre silencieusement les donnees
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    for b in range(N_BUCKETS):
        os.makedirs(os.path.join(TMP_DIR, f"bucket_{b:02d}"), exist_ok=True)

    stats = {c: WelfordStats() for c in smart_cols}
    rows_total = 0
    chunk_idx = 0

    for chunk in pd.read_csv(in_path, chunksize=CHUNKSIZE):
        rows_total += len(chunk)
        chunk_idx += 1

        desc = chunk[smart_cols].agg(["count", "mean", "var"])
        for c in smart_cols:
            stats[c].merge_chunk(
                int(desc.at["count", c]),
                float(desc.at["mean", c]) if not math.isnan(desc.at["mean", c]) else 0.0,
                float(desc.at["var", c]) if not math.isnan(desc.at["var", c]) else 0.0,
            )

        chunk = chunk.assign(_bucket=chunk["serial_number"].map(bucket_of))
        for b, g in chunk.groupby("_bucket", sort=False):
            g.drop(columns="_bucket").to_parquet(
                os.path.join(TMP_DIR, f"bucket_{b:02d}", f"part_{chunk_idx:05d}.parquet"),
                index=False,
            )

        print(f"  ... {rows_total} lignes partitionnees")

    return rows_total, stats


def pass2_clean(model, smart_cols, dropped_constant, out_path):
    header_written = False
    rows_before = 0
    rows_after = 0
    disks_before = set()
    disks_after = set()
    n_dup_total = 0
    n_short_disks = 0
    n_short_rows = 0
    n_incoherent = 0
    incoherent_serials = []
    failures_after = 0
    n_all_na_total = 0
    n_remaining_na = 0
    kept_smart = [c for c in smart_cols if c not in dropped_constant]

    with open(out_path, "w", encoding="utf-8", newline="") as out_f:
        for b in range(N_BUCKETS):
            bucket_dir = os.path.join(TMP_DIR, f"bucket_{b:02d}")
            if not os.path.isdir(bucket_dir) or not os.listdir(bucket_dir):
                continue

            df = pd.read_parquet(bucket_dir)
            rows_before += len(df)
            disks_before.update(df["serial_number"].unique())

            df = df.drop(columns=[c for c in dropped_constant if c in df.columns])
            # certains fichiers Backblaze sources utilisent un format de date
            # non-standard (ex. "2/25/18" au lieu de "2018-02-25") -> format="mixed"
            df["date"] = pd.to_datetime(df["date"], format="mixed")
            df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)

            dup_mask = df.duplicated(subset=["serial_number", "date"], keep="first")
            n_dup_total += int(dup_mask.sum())
            df = df[~dup_mask]

            grp = df.groupby("serial_number", sort=False)
            df[kept_smart] = grp[kept_smart].ffill()
            df[kept_smart] = df.groupby("serial_number", sort=False)[kept_smart].bfill()
            # on ne retire une ligne que si TOUTES les colonnes SMART gardees sont
            # manquantes (aucune info exploitable). Un NaN partiel est legitime :
            # certains attributs n'existent que pour certaines generations de
            # disques (firmware) -> laisse tel quel, a gerer lors du pretraitement
            all_na = df[kept_smart].isna().all(axis=1)
            n_all_na_total += int(all_na.sum())
            df = df[~all_na]
            n_remaining_na += int(df[kept_smart].isna().sum().sum())

            history_len = df.groupby("serial_number")["date"].transform("count")
            short_mask = history_len < MIN_HISTORY
            n_short_disks += df.loc[short_mask, "serial_number"].nunique()
            n_short_rows += int(short_mask.sum())
            df = df[~short_mask]

            if not df.empty:
                # coherent = 0 panne, OU exactement 1 panne situee sur la derniere
                # ligne du disque (deja trie par serial_number puis date)
                is_last_row = ~df["serial_number"].duplicated(keep="last")
                total_failures = df.groupby("serial_number", sort=False)["failure"].transform("sum")
                failure_not_last = (df["failure"] == 1) & ~is_last_row
                bad_row = (total_failures > 1) | failure_not_last
                bad = df.loc[bad_row, "serial_number"].unique().tolist()
                n_incoherent += len(bad)
                incoherent_serials.extend(bad)

            df.to_csv(out_f, index=False, header=not header_written)
            header_written = True

            rows_after += len(df)
            disks_after.update(df["serial_number"].unique())
            failures_after += int(df["failure"].sum())

            print(f"  ... bucket {b + 1}/{N_BUCKETS} nettoye ({len(df)} lignes conservees)")
            del df, grp
            gc.collect()

    return {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "disks_before": len(disks_before),
        "disks_after": len(disks_after),
        "n_dup": n_dup_total,
        "n_short_disks": n_short_disks,
        "n_short_rows": n_short_rows,
        "n_incoherent": n_incoherent,
        "incoherent_serials": incoherent_serials,
        "failures_after": failures_after,
        "kept_smart": kept_smart,
        "n_all_na_dropped": n_all_na_total,
        "n_remaining_na": n_remaining_na,
    }


def clean(model):
    in_path = os.path.join(ISOLATED_DIR, f"isolated_{model}.csv")
    if not os.path.exists(in_path):
        raise SystemExit(f"{in_path} introuvable. Lance d'abord scripts/01_isolate.py.")

    header = pd.read_csv(in_path, nrows=0).columns.tolist()
    smart_cols = [c for c in header if c.startswith("smart_")]

    print(f"Passe 1/2 : partitionnement en {N_BUCKETS} buckets + statistiques en ligne...")
    rows_total, stats = pass1_partition(in_path, smart_cols)

    dropped_constant = [c for c in smart_cols if stats[c].std() <= STD_EPSILON]
    dropped_sparse = [
        c for c in smart_cols
        if c not in dropped_constant and (1 - stats[c].n / rows_total) > MAX_MISSING_RATE
    ]
    dropped_cols = dropped_constant + dropped_sparse
    print(
        f"\n{len(dropped_constant)} colonne(s) constante(s) (ecart-type ~ 0) + "
        f"{len(dropped_sparse)} colonne(s) trop peu remplie(s) (> {MAX_MISSING_RATE * 100:.0f}% "
        f"manquant) -> {len(dropped_cols)} colonne(s) retiree(s) au total.\n"
    )

    os.makedirs(CLEAN_DIR, exist_ok=True)
    out_path = os.path.join(CLEAN_DIR, f"clean_{model}.csv")

    print("Passe 2/2 : nettoyage par bucket (tri, doublons, trous, historique court)...")
    result = pass2_clean(model, smart_cols, dropped_cols, out_path)

    shutil.rmtree(TMP_DIR, ignore_errors=True)

    os.makedirs("reports", exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(f"# Rapport de nettoyage - modele {model}\n\n")
        f.write(f"- Lignes en entree : {result['rows_before']}\n")
        f.write(
            f"- Lignes en sortie : {result['rows_after']} "
            f"({result['rows_before'] - result['rows_after']} retirees, "
            f"{(1 - result['rows_after'] / result['rows_before']) * 100:.1f}%)\n"
        )
        f.write(f"- Disques en entree : {result['disks_before']}\n")
        f.write(f"- Disques en sortie : {result['disks_after']}\n")
        f.write(f"- Pannes conservees : {result['failures_after']}\n\n")

        f.write(f"## Colonnes SMART constantes retirees ({len(dropped_constant)})\n")
        f.write("Critere : ecart-type ~ 0 sur l'ensemble du dataset isole "
                 "(calcule en ligne, algorithme de Welford) -> aucune information utile.\n\n")
        for c in sorted(dropped_constant):
            f.write(f"- {c}\n")
        f.write(f"\n## Colonnes SMART trop peu remplies retirees ({len(dropped_sparse)})\n")
        f.write(
            f"Critere : plus de {MAX_MISSING_RATE * 100:.0f}% de valeurs manquantes sur "
            f"l'ensemble du dataset isole. Ces attributs SMART ne sont rapportes que par "
            f"certaines generations de disques (firmware different selon l'annee de "
            f"fabrication) -> pas assez fiables pour etre exploites.\n\n"
        )
        for c in sorted(dropped_sparse):
            f.write(f"- {c}\n")
        f.write(f"\n## Colonnes SMART conservees ({len(result['kept_smart'])})\n\n")
        for c in sorted(result["kept_smart"]):
            f.write(f"- {c}\n")

        f.write(f"\n## Doublons\n- {result['n_dup']} ligne(s) dupliquee(s) "
                f"(meme disque + meme date) supprimee(s)\n")

        f.write(
            f"\n## Valeurs manquantes restantes\n"
            f"Les trous ponctuels sont combles par disque (ffill/bfill, jamais entre deux "
            f"disques). Une ligne n'est retiree que si TOUTES les colonnes SMART gardees "
            f"sont manquantes (aucune info exploitable) : {result['n_all_na_dropped']} "
            f"ligne(s) dans ce cas, supprimee(s).\n"
            f"- Valeurs encore manquantes (NaN partiel, legitime) dans le fichier final : "
            f"{result['n_remaining_na']} -> a gerer explicitement lors du pretraitement "
            f"(ex. imputation ou exclusion selon la colonne) si necessaire pour le modele.\n"
        )

        f.write(
            f"\n## Disques a historique trop court\n"
            f"- Seuil : moins de {MIN_HISTORY} jours d'historique\n"
            f"- {result['n_short_disks']} disque(s) ecarte(s), "
            f"soit {result['n_short_rows']} ligne(s)\n"
        )

        f.write(
            f"\n## Coherence de la colonne failure\n"
            f"- {result['n_incoherent']} disque(s) avec une anomalie (failure=1 pas sur la "
            f"derniere ligne, ou plusieurs failure=1) -> a inspecter manuellement\n"
        )
        if result["incoherent_serials"]:
            shown = result["incoherent_serials"][:20]
            more = " ..." if len(result["incoherent_serials"]) > 20 else ""
            f.write(f"- Disques concernes : {shown}{more}\n")

    print(f"\nFichier propre ecrit : {out_path}")
    print(f"Rapport ecrit        : {OUT_REPORT}")
    print(
        f"\nResume : {result['rows_before']} -> {result['rows_after']} lignes | "
        f"{result['disks_before']} -> {result['disks_after']} disques | "
        f"{result['failures_after']} pannes conservees | "
        f"{len(result['kept_smart'])} colonnes SMART (sur {len(smart_cols)})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Nom exact du modele (meme que pour 01_isolate.py)")
    args = parser.parse_args()
    clean(args.model)
