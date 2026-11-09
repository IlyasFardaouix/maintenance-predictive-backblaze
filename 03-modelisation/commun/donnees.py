"""Téléchargement et chargement économe en mémoire d'un split du dataset prétraité.

Applique les corrections du protocole commun :
  1. suppression des lignes qui suivent la première RUL = 0 de chaque disque ;
  2. découpage en segments seulement si l'écart de dates dépasse MAX_GAP_DAYS.
"""
import gc
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from . import protocole as P


def download_parquet(repo_id=P.HF_DATASET_REPO, file_name=P.HF_DATASET_FILE):
    """Télécharge le fichier prétraité depuis Hugging Face (mis en cache) et renvoie son chemin."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo_id, repo_type="dataset")
    target = [f for f in files if f.endswith(file_name)]
    if not target:
        raise FileNotFoundError(f"{file_name} introuvable dans {repo_id} : {files}")
    return hf_hub_download(repo_id, target[0], repo_type="dataset")


def list_features(path):
    """Les 24 colonnes smart_*_raw, triées par numéro d'attribut."""
    names = pq.ParquetFile(path).schema_arrow.names
    for col in ["serial_number", "date", "rul", "split"]:
        if col not in names:
            raise KeyError(f"Colonne manquante : {col}")
    return sorted([c for c in names if c.startswith("smart_")], key=lambda c: int(c.split("_")[1]))


def split_names(path):
    """Noms réels des splits train / validation / test dans le fichier."""
    names = pl.scan_parquet(path).select(pl.col("split").unique()).collect()["split"].to_list()

    def find(key):
        m = [s for s in names if key in str(s).lower()]
        if not m:
            raise ValueError(f"Split '{key}' introuvable dans {names}")
        return m[0]

    return find("train"), find("val"), find("test")


def load_split(path, name, features, store_dtype=np.float16, max_gap_days=P.MAX_GAP_DAYS, verbose=True):
    """Charge un split et renvoie un dict de tableaux numpy.

    X          : (n, n_features) capteurs normalisés (float16 par défaut)
    rul        : (n,) RUL en jours, bornée à [0, RUL_CAP]
    pos        : (n,) position de la ligne dans son segment continu
    seg        : (n,) identifiant de segment
    date       : (n,) jours depuis 1970
    seg_serial : liste, numéro de série de chaque segment
    """
    t0 = time.time()
    df = (pl.scan_parquet(path)
            .filter(pl.col("split") == name)
            .select([pl.col("serial_number"), pl.col("date"), pl.col("rul").cast(pl.Float32)]
                    + [pl.col(c).cast(pl.Float32) for c in features])
            .collect())
    dt = df.schema["date"]
    if dt == pl.Utf8:
        df = df.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
    elif isinstance(dt, pl.Datetime):
        df = df.with_columns(pl.col("date").dt.date())
    df = df.sort(["serial_number", "date"])
    n0 = len(df)

    # Règle 1 : supprimer les lignes après la première RUL = 0 de chaque disque
    new_serial = df.select((pl.col("serial_number") != pl.col("serial_number").shift(1))
                           .fill_null(True).alias("b"))["b"].to_numpy()
    rul_all = df["rul"].to_numpy()
    idx = np.arange(n0, dtype=np.int64)
    first = np.maximum.accumulate(np.where(new_serial, idx, 0))
    z = (rul_all == 0).astype(np.int64)
    zb = np.cumsum(z) - z                       # nombre de RUL = 0 avant la ligne
    keep = (zb - zb[first]) == 0                # aucune RUL = 0 avant, dans ce disque
    n_zero_rows, n_zero_disks = int(z.sum()), int(np.unique(first[z == 1]).size)
    del idx, first, z, zb

    # Règle 2 : couper la série seulement si l'écart de dates dépasse max_gap_days
    kept_idx = np.flatnonzero(keep)
    date = df["date"].to_physical().to_numpy()[kept_idx].astype(np.int32)
    new_serial = new_serial[kept_idx]
    gap = np.ones(len(date), dtype=bool)
    gap[1:] = np.diff(date) > max_gap_days
    brk = new_serial | gap

    n = len(date)
    idx = np.arange(n, dtype=np.int64)
    starts = np.maximum.accumulate(np.where(brk, idx, 0))
    pos = (idx - starts).astype(np.int32)
    seg = (np.cumsum(brk) - 1).astype(np.int32)
    del idx, starts

    X = np.empty((n, len(features)), dtype=store_dtype)
    for j, c in enumerate(features):
        X[:, j] = df[c].fill_null(0).fill_nan(0).to_numpy()[kept_idx]
    rul = np.clip(rul_all[kept_idx], 0, P.RUL_CAP).astype(np.float32)
    seg_serial = df["serial_number"].gather(kept_idx[brk]).to_list()
    del df, rul_all, keep, kept_idx, brk, gap, new_serial
    gc.collect()

    if verbose:
        print(f"{name:>5} : {n0:>11,} lignes | RUL=0 : {n_zero_rows:,} lignes sur {n_zero_disks:,} disques "
              f"-> {n0 - n:,} supprimées | {len(seg_serial):,} segments pour {len(set(seg_serial)):,} disques "
              f"| X = {X.nbytes / 1e9:.2f} Go | {time.time() - t0:.0f} s")
    return dict(name=name, X=X, rul=rul, pos=pos, seg=seg, date=date, seg_serial=seg_serial)
