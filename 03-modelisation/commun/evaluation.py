"""Métriques communes du protocole : régression RUL, alerte par fenêtre, alerte par disque.

Toutes les fonctions prennent des tableaux numpy de RUL réelles (y) et prédites (p), en jours.
"""
import os

import numpy as np
import polars as pl
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)

from . import protocole as P


def rmse(y, p):
    return float(np.sqrt(np.mean((p - y) ** 2))) if len(y) else float("nan")


def nasa_score(y, p):
    """Score asymétrique : une RUL surestimée (alerte tardive) est plus pénalisée."""
    d = p - y
    return float(np.mean(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def reg(y, p):
    return dict(n=int(len(y)), RMSE=rmse(y, p),
                MAE=float(np.mean(np.abs(p - y))) if len(y) else float("nan"))


def selection_criterion(y, p):
    """Critère commun de sélection : moyenne du RMSE en dégradation et du RMSE sur les sains."""
    low = y < P.RUL_CAP
    r_low, r_high = rmse(y[low], p[low]), rmse(y[~low], p[~low])
    return 0.5 * (r_low + r_high), r_low, r_high


def choose_threshold(y_val, p_val):
    """Seuil d'alerte qui maximise le F1 sur TOUTE la validation (score = -RUL prédite)."""
    y_alert = (y_val <= P.ALERT_HORIZON).astype(np.int8)
    prec, rec, thr = precision_recall_curve(y_alert, -p_val)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(-thr[i]), dict(precision=float(prec[i]), rappel=float(rec[i]), F1=float(f1[i]))


def evaluate(y, p, seg, seg_serial, threshold):
    """Toutes les métriques du protocole sur un split complet (vraie distribution)."""
    low = y < P.RUL_CAP
    imm = y <= P.ALERT_HORIZON
    res = {"regression": {
        "global (vraie distribution)": reg(y, p),
        "degradation (RUL<125)": reg(y[low], p[low]),
        f"imminent (RUL<={P.ALERT_HORIZON})": reg(y[imm], p[imm]),
        "baseline 'toujours 125' (RUL<125)": reg(y[low], np.full(low.sum(), P.RUL_CAP, np.float32)),
    }}
    res["regression"]["degradation (RUL<125)"]["score_NASA_moyen"] = nasa_score(y[low], p[low])

    bins = [(0, 10), (10, 30), (30, 60), (60, 125), (125, 126)]
    res["par_tranche"] = {(f"[{a},{b})" if b <= P.RUL_CAP else "125 (sain)"):
                          reg(y[(y >= a) & (y < b)], p[(y >= a) & (y < b)]) for a, b in bins}

    y_alert = imm.astype(np.int8)
    p_alert = (p <= threshold).astype(np.int8)
    res["alerte_fenetre"] = dict(
        seuil=float(threshold),
        precision=float(precision_score(y_alert, p_alert, zero_division=0)),
        rappel=float(recall_score(y_alert, p_alert, zero_division=0)),
        F1=float(f1_score(y_alert, p_alert, zero_division=0)),
        AP=float(average_precision_score(y_alert, -p)),
        matrice_confusion=confusion_matrix(y_alert, p_alert, labels=[0, 1]).tolist())

    # Niveau disque : par numéro de série
    disk_names, seg2disk = np.unique(np.array(seg_serial), return_inverse=True)
    disk = seg2disk[seg]
    n_disk = len(disk_names)
    failing = np.bincount(disk, weights=low, minlength=n_disk) > 0
    has_win = np.bincount(disk, minlength=n_disk) > 0
    alarm_imm = np.bincount(disk, weights=(p_alert & y_alert), minlength=n_disk) > 0
    alarm_any = np.bincount(disk, weights=p_alert, minlength=n_disk) > 0
    first_alarm_rul = np.full(n_disk, -1.0)
    m = (p_alert == 1) & low
    np.maximum.at(first_alarm_rul, disk[m], y[m])          # RUL réelle à la 1re alerte
    healthy = has_win & ~failing
    detected = failing & alarm_imm
    res["alerte_disque"] = dict(
        disques_defaillants=int(failing.sum()),
        detectes_avant_panne=int(detected.sum()),
        taux_detection=float(detected.sum() / max(failing.sum(), 1)),
        avance_moyenne_jours=float(first_alarm_rul[detected].mean()) if detected.any() else None,
        disques_sains=int(healthy.sum()),
        taux_fausse_alerte_disques_sains=float((alarm_any & healthy).sum() / max(healthy.sum(), 1)),
    )
    return res, dict(disk=disk, disk_names=disk_names, failing=failing)


def save_predictions(d, ends, pred, path):
    """Fichier de prédictions au format commun : serial_number, date, rul_true, rul_pred."""
    seg_tab = pl.DataFrame({"seg": np.arange(len(d["seg_serial"]), dtype=np.int32),
                            "serial_number": d["seg_serial"]})
    (pl.DataFrame({"seg": d["seg"][ends], "date": d["date"][ends],
                   "rul_true": d["rul"][ends], "rul_pred": pred.astype(np.float32)})
       .with_columns(pl.col("date").cast(pl.Date))
       .join(seg_tab, on="seg", how="left")
       .select(["serial_number", "date", "rul_true", "rul_pred"])
       .write_parquet(path))
    print(f"{os.path.basename(path)} : {len(ends):,} lignes")
