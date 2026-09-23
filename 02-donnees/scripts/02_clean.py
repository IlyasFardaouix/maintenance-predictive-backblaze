"""
Etape 2 - Nettoyage (cleaning) du dataset isole.

Ce script ne fait PAS encore le pretraitement final pour le LSTM (pas de calcul
de RUL, pas de normalisation, pas de fenetrage : ça viendra dans une etape 3
"preprocessing" separee, une fois le cleaning valide). Ici on se contente de
rendre les donnees fiables et exploitables :

  1. Parsing des dates + tri par disque puis par date
  2. Suppression des doublons exacts (meme disque, meme date)
  3. Suppression des colonnes constantes / quasi constantes (aucune info utile)
  4. Remplissage des trous ponctuels PAR DISQUE (forward/backward fill) -
     jamais entre deux disques differents, pour ne pas introduire de fuite
  5. Suppression des disques avec trop peu d'historique (seuil MIN_HISTORY)
  6. Verification de coherence : une fois failure=1 atteint, c'est la derniere
     ligne du disque (sinon on logue l'anomalie sans la corriger a l'aveugle)

Usage :
    .venv/Scripts/python.exe scripts/02_clean.py --model ST4000DM000

Entree  : data/isolated/isolated_<model>.csv
Sortie  : data/clean/clean_<model>.csv
          reports/02_cleaning_report.md (justification des choix, pour le
          livrable L1 du cahier des charges)
"""

import argparse
import os

import pandas as pd

ISOLATED_DIR = os.path.join("data", "isolated")
CLEAN_DIR = os.path.join("data", "clean")
OUT_REPORT = os.path.join("reports", "02_cleaning_report.md")

MIN_HISTORY = 30  # un disque avec moins de 30 jours de donnees est ecarte
QUASI_CONSTANT_THRESHOLD = 0.99  # colonne ecartee si >99% des valeurs identiques


def clean(model):
    in_path = os.path.join(ISOLATED_DIR, f"isolated_{model}.csv")
    if not os.path.exists(in_path):
        raise SystemExit(f"{in_path} introuvable. Lance d'abord scripts/01_isolate.py.")

    df = pd.read_csv(in_path)
    rows_before = len(df)
    disks_before = df["serial_number"].nunique()

    log = [f"# Rapport de nettoyage - modele {model}\n"]
    log.append(f"- Lignes en entree : {rows_before}")
    log.append(f"- Disques en entree : {disks_before}\n")

    # 1. dates + tri
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)

    # 2. doublons exacts (meme disque, meme jour)
    dup_mask = df.duplicated(subset=["serial_number", "date"], keep="first")
    n_dup = int(dup_mask.sum())
    df = df[~dup_mask]
    log.append(f"## Doublons\n- {n_dup} ligne(s) dupliquee(s) (meme disque + meme date) supprimee(s)\n")

    # 3. colonnes constantes / quasi constantes
    smart_cols = [c for c in df.columns if c.startswith("smart_")]
    dropped_constant = []
    for c in smart_cols:
        top_share = df[c].value_counts(normalize=True, dropna=False).iloc[0] if df[c].notna().any() else 1.0
        if top_share >= QUASI_CONSTANT_THRESHOLD:
            dropped_constant.append(c)
    df = df.drop(columns=dropped_constant)
    remaining_smart = [c for c in smart_cols if c not in dropped_constant]
    log.append(
        f"## Colonnes constantes / quasi constantes retirees ({len(dropped_constant)})\n"
        f"Seuil : >= {QUASI_CONSTANT_THRESHOLD * 100:.0f}% de valeurs identiques -> aucune "
        f"information utile pour le modele.\n"
    )
    for c in dropped_constant:
        log.append(f"- {c}")
    log.append("")

    # 4. remplissage des trous PAR DISQUE uniquement (pas de fuite entre disques)
    missing_before = df[remaining_smart].isna().sum().sum()
    df[remaining_smart] = df.groupby("serial_number")[remaining_smart].transform(
        lambda s: s.ffill().bfill()
    )
    missing_after = df[remaining_smart].isna().sum().sum()
    log.append(
        f"## Valeurs manquantes\n"
        f"- Avant remplissage (ffill/bfill par disque) : {int(missing_before)} valeurs manquantes\n"
        f"- Apres : {int(missing_after)} valeurs manquantes restantes "
        f"(disques n'ayant aucune valeur exploitable sur ce capteur)\n"
    )
    # lignes encore incompletes -> on les retire (rares, disque sans aucune mesure valide)
    rows_with_na = df[remaining_smart].isna().any(axis=1)
    n_rows_na = int(rows_with_na.sum())
    df = df[~rows_with_na]
    log.append(f"- {n_rows_na} ligne(s) encore incomplete(s) apres remplissage -> supprimee(s)\n")

    # 5. disques avec trop peu d'historique
    history_len = df.groupby("serial_number")["date"].transform("count")
    short_mask = history_len < MIN_HISTORY
    n_short_disks = df.loc[short_mask, "serial_number"].nunique()
    n_short_rows = int(short_mask.sum())
    df = df[~short_mask]
    log.append(
        f"## Disques a historique trop court\n"
        f"- Seuil : moins de {MIN_HISTORY} jours d'historique (necessaire pour "
        f"le fenetrage a venir en pretraitement)\n"
        f"- {n_short_disks} disque(s) ecarte(s), soit {n_short_rows} ligne(s)\n"
    )

    # 6. coherence failure : doit etre la derniere ligne du disque si presente
    def last_row_is_failure_or_none(g):
        if g["failure"].sum() == 0:
            return True
        return g.iloc[-1]["failure"] == 1 and g["failure"].sum() == 1

    incoherent = df.groupby("serial_number").apply(last_row_is_failure_or_none, include_groups=False)
    n_incoherent = int((~incoherent).sum())
    log.append(
        f"## Coherence de la colonne failure\n"
        f"- {n_incoherent} disque(s) avec une anomalie (failure=1 pas sur la derniere ligne, "
        f"ou plusieurs failure=1) -> a inspecter manuellement, non corrige automatiquement\n"
    )
    if n_incoherent > 0:
        bad_serials = incoherent[~incoherent].index.tolist()
        log.append(f"- Disques concernes : {bad_serials[:20]}"
                    f"{' ...' if len(bad_serials) > 20 else ''}\n")

    rows_after = len(df)
    disks_after = df["serial_number"].nunique()
    failures_after = int(df["failure"].sum())

    os.makedirs(CLEAN_DIR, exist_ok=True)
    out_path = os.path.join(CLEAN_DIR, f"clean_{model}.csv")
    df.to_csv(out_path, index=False)

    log.insert(
        1,
        f"- Lignes en sortie : {rows_after} ({rows_before - rows_after} retirees, "
        f"{(1 - rows_after / rows_before) * 100:.1f}%)\n"
        f"- Disques en sortie : {disks_after} ({disks_before - disks_after} retires)\n"
        f"- Pannes conservees : {failures_after}\n"
        f"- Colonnes SMART conservees : {len(remaining_smart)}\n",
    )

    os.makedirs("reports", exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(log))

    print(f"Fichier propre ecrit : {out_path}")
    print(f"Rapport ecrit        : {OUT_REPORT}")
    print(
        f"\nResume : {rows_before} -> {rows_after} lignes | "
        f"{disks_before} -> {disks_after} disques | {failures_after} pannes conservees | "
        f"{len(remaining_smart)} colonnes SMART"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Nom exact du modele (meme que pour 01_isolate.py)")
    args = parser.parse_args()
    clean(args.model)
