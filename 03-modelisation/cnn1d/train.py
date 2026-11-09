"""Entraînement et évaluation du CNN-1D selon le protocole commun.

Usage (Kaggle, GPU T4) :
    python 03-modelisation/cnn1d/train.py --out /kaggle/working/cnn1d_outputs

Sorties dans --out :
    cnn1d_best.pt, config_cnn1d.json, history_cnn1d.csv, metrics_test_cnn1d.json,
    predictions_val_cnn1d.parquet, predictions_test_cnn1d.parquet, figures/*.png
"""
import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # rend 'commun' importable
from commun import protocole as P                              # noqa: E402
from commun.donnees import download_parquet, list_features, load_split, split_names  # noqa: E402
from commun.evaluation import (choose_threshold, evaluate, rmse, save_predictions,     # noqa: E402
                               selection_criterion)
from commun.fenetres import WindowStore, sample_ends, window_ends  # noqa: E402
from config import CFG                                           # noqa: E402
from model import CNN1D                                          # noqa: E402


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=os.environ.get("PDM_PARQUET"),
                    help="chemin local du fichier prétraité (sinon téléchargement Hugging Face)")
    ap.add_argument("--out", default="cnn1d_outputs")
    ap.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    args = ap.parse_args()
    out = Path(args.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    seed_all(P.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device)

    # 1. Données
    path = args.parquet or download_parquet()
    features = list_features(path)
    s_train, s_val, s_test = split_names(path)
    store = np.float16 if CFG.STORE_DTYPE == "float16" else np.float32
    tr, va, te = (load_split(path, s, features, store) for s in (s_train, s_val, s_test))

    # 2. Fenêtres et rééquilibrage
    for d in (tr, va, te):
        d["ends_all"], d["ends_low"], d["ends_high"] = window_ends(d)
        print(f"{d['name']:>5} : {len(d['ends_all']):,} fenêtres dont {len(d['ends_low']):,} en dégradation")
    val_ends_s = sample_ends(va["ends_low"], va["ends_high"], P.HEALTHY_RATIO, np.random.default_rng(P.SEED))

    # 3. Séries sur le GPU si elles tiennent dans 60 % de la VRAM
    total = sum(d["X"].nbytes for d in (tr, va, te))
    on_gpu = device.type == "cuda" and total < 0.6 * torch.cuda.get_device_properties(0).total_memory
    data_dev = device if on_gpu else torch.device("cpu")
    W = {d["name"]: WindowStore(d, device, data_dev) for d in (tr, va, te)}
    for d in (tr, va, te):
        del d["X"]
    gc.collect()

    # 4. Modèle et entraînement
    model = CNN1D(len(features), CFG.CHANNELS, CFG.KERNEL, CFG.DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = lambda: torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    loss_fn = nn.HuberLoss(delta=CFG.HUBER_DELTA)
    ckpt = out / "cnn1d_best.pt"

    @torch.no_grad()
    def predict(store_, ends):
        model.eval()
        res = np.empty(len(ends), dtype=np.float32)
        for i in range(0, len(ends), CFG.EVAL_BATCH):
            xb, _ = store_.batch(ends[i:i + CFG.EVAL_BATCH])
            with autocast():
                p = model(xb)
            res[i:i + CFG.EVAL_BATCH] = (p.float().clamp(0, 1) * P.RUL_CAP).cpu().numpy()
        return res

    rng = np.random.default_rng(P.SEED)
    yv_s = va["rul"][val_ends_s]
    history, best, bad, t0 = [], float("inf"), 0, time.time()
    for ep in range(1, args.epochs + 1):
        te0 = time.time()
        model.train()
        ends = sample_ends(tr["ends_low"], tr["ends_high"], P.HEALTHY_RATIO, rng)
        run, nb = 0.0, 0
        for i in range(0, len(ends), CFG.BATCH_SIZE):
            xb, yb = W[tr["name"]].batch(ends[i:i + CFG.BATCH_SIZE])
            opt.zero_grad(set_to_none=True)
            with autocast():
                o = model(xb)
            loss = loss_fn(o.float(), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            run += loss.item(); nb += 1

        pv = predict(W[va["name"]], val_ends_s)
        crit, v_low, v_high = selection_criterion(yv_s, pv)
        sched.step(crit)
        history.append(dict(epoch=ep, train_loss=run / nb, val_rmse=rmse(yv_s, pv), val_rmse_low=v_low,
                            val_rmse_high=v_high, crit=crit, lr=opt.param_groups[0]["lr"]))
        flag = ""
        if crit < best - 1e-3:
            best, bad, flag = crit, 0, "  <- meilleur, sauvegardé"
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        print(f"Epoch {ep:02d} | loss {run / nb:.4f} | RMSE dégr. {v_low:6.2f} j | RMSE sain {v_high:6.2f} j "
              f"| critère {crit:6.2f} | {time.time() - te0:.0f} s{flag}")
        if bad >= CFG.PATIENCE:
            print("Early stopping."); break
        if (time.time() - t0) / 60 > CFG.MAX_TRAIN_MINUTES:
            print("Limite de temps atteinte."); break
    train_min = (time.time() - t0) / 60
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"Meilleur critère : {best:.2f} jours | durée : {train_min:.1f} min")

    # 5. Seuil d'alerte sur toute la validation, puis évaluation sur tout le test
    pv_all = predict(W[va["name"]], va["ends_all"])
    thresh, val_alert = choose_threshold(va["rul"][va["ends_all"]], pv_all)
    print(f"Seuil retenu : {thresh:.1f} jours | validation : {val_alert}")
    t_pred = time.time()
    pt = predict(W[te["name"]], te["ends_all"])
    pred_seconds = time.time() - t_pred
    yt, seg_t = te["rul"][te["ends_all"]], te["seg"][te["ends_all"]]
    results, disk_info = evaluate(yt, pt, seg_t, te["seg_serial"], thresh)
    results["temps_prediction_test_s"] = round(pred_seconds, 2)
    print(json.dumps(results, indent=2, ensure_ascii=False))

    # 6. Sauvegardes
    save_predictions(va, va["ends_all"], pv_all, out / "predictions_val_cnn1d.parquet")
    save_predictions(te, te["ends_all"], pt, out / "predictions_test_cnn1d.parquet")
    cfg = {k: v for k, v in vars(CFG).items()}
    cfg.update(protocole={k: getattr(P, k) for k in dir(P) if k.isupper()}, FEATURES=features,
               version="v2", n_parametres=n_params, epochs_effectues=len(history),
               temps_entrainement_min=round(train_min, 2), meilleur_critere_val=best, seuil_alerte=thresh,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    (out / "config_cnn1d.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False, default=str))
    (out / "metrics_test_cnn1d.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    pl.DataFrame(history).write_csv(out / "history_cnn1d.csv")
    save_figures(history, yt, pt, results, disk_info, seg_t, thresh, out / "figures")
    print("Fichiers sauvegardés dans", out)


def save_figures(history, yt, pt, results, disk_info, seg_t, thresh, fig_dir):
    h = pl.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    ax[0].plot(h["epoch"], h["train_loss"], marker="o"); ax[0].set_title("Perte Huber (train)")
    ax[1].plot(h["epoch"], h["val_rmse_low"], marker="o", label="RMSE dégradation")
    ax[1].plot(h["epoch"], h["val_rmse_high"], marker="o", label="RMSE sain")
    ax[1].plot(h["epoch"], h["crit"], marker="o", lw=2.5, label="critère (moyenne)")
    ax[1].set_title("Validation (jours)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(fig_dir / "courbes_apprentissage.png", dpi=120); plt.close(fig)

    low = yt < P.RUL_CAP
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(low)
    sel = rng.choice(idx, min(50_000, len(idx)), replace=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].scatter(yt[sel], pt[sel], s=2, alpha=0.2); ax[0].plot([0, P.RUL_CAP], [0, P.RUL_CAP], "r--")
    ax[0].set_xlabel("RUL réelle"); ax[0].set_ylabel("RUL prédite"); ax[0].set_title("Prédit vs réel (dégradation)")
    names = list(results["par_tranche"].keys())
    ax[1].bar(names, [results["par_tranche"][k]["MAE"] for k in names])
    ax[1].set_title("MAE par tranche de RUL réelle (jours)")
    fig.tight_layout(); fig.savefig(fig_dir / "resultats_test.png", dpi=120); plt.close(fig)

    disk, disk_names, failing = disk_info["disk"], disk_info["disk_names"], disk_info["failing"]
    fail_ids = np.flatnonzero(failing)
    show = rng.choice(fail_ids, min(6, len(fail_ids)), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    for a, s in zip(axes.ravel(), show):
        mk = disk == s
        yy, pp = yt[mk][-200:], pt[mk][-200:]
        x = np.arange(-len(yy) + 1, 1)
        a.plot(x, yy, label="RUL réelle"); a.plot(x, pp, label="RUL prédite", alpha=0.8)
        a.axhline(thresh, color="red", ls="--", lw=1, label="seuil")
        a.set_title(disk_names[s], fontsize=9)
    axes.ravel()[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "disques_test.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
