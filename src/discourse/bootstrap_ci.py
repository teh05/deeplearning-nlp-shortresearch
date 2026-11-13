# -*- coding: utf-8 -*-
"""Paired bootstrap CI on test predictions (resample text_id, not seeds)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from discourse.calibration import brier_score, expected_calibration_error  # noqa: E402
from discourse.utils import ensure_dir  # noqa: E402


def load_preds(exp_id: str, seed: int) -> pd.DataFrame:
    path = ROOT / "artifacts" / "predictions" / "seeded" / f"{exp_id}_seed{seed}_test_preds.csv"
    if not path.exists():
        # fallback non-seeded legacy
        alt = ROOT / "artifacts" / "predictions" / f"{exp_id}_test_preds.csv"
        if alt.exists() and seed == 42:
            df = pd.read_csv(alt)
            df["seed"] = 42
            return df
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def task_scores(df: pd.DataFrame, task: str, threshold: float) -> dict:
    hard = f"hard_{task}"
    prob = f"prob_{task}"
    sub = df[df[hard].notna()].copy()
    y = sub[hard].astype(int).to_numpy()
    p = sub[prob].to_numpy()
    pred = (p >= threshold).astype(int)
    out = {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "pos_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "ece": float(expected_calibration_error(y, p)),
        "brier": float(brier_score(y, p)),
    }
    return out


def paired_bootstrap_delta(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    task: str,
    thr_a: float,
    thr_b: float,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Bootstrap over shared text_ids; delta = B - A."""
    hard = f"hard_{task}"
    a = df_a[df_a[hard].notna()][["text_id", f"prob_{task}", hard]].rename(
        columns={f"prob_{task}": "prob_a", hard: "y"}
    )
    b = df_b[df_b[hard].notna()][["text_id", f"prob_{task}"]].rename(columns={f"prob_{task}": "prob_b"})
    m = a.merge(b, on="text_id", how="inner")
    if len(m) == 0:
        return {"n": 0}
    y = m["y"].astype(int).to_numpy()
    pa = m["prob_a"].to_numpy()
    pb = m["prob_b"].to_numpy()
    ids = np.arange(len(m))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(ids, size=len(ids), replace=True)
        ya, paa, pbb = y[idx], pa[idx], pb[idx]
        fa = f1_score(ya, (paa >= thr_a).astype(int), average="macro", zero_division=0)
        fb = f1_score(ya, (pbb >= thr_b).astype(int), average="macro", zero_division=0)
        deltas.append(fb - fa)
    deltas = np.asarray(deltas, dtype=float)
    return {
        "n": int(len(m)),
        "metric": "macro_f1",
        "mean_delta": float(deltas.mean()),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "n_boot": n_boot,
    }


def threshold_from_metrics(exp_id: str, seed: int, task: str) -> float:
    path = ROOT / "artifacts" / "metrics" / "seeded" / f"{exp_id}_seed{seed}_test_metrics.csv"
    if path.exists():
        df = pd.read_csv(path)
        hit = df[df.task == task]
        if len(hit):
            return float(hit.iloc[0]["threshold"])
    return 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42, help="Which model seed's predictions to use")
    args = parser.parse_args()

    pairs = [
        ("E1c_indobertweet", "E2_stl"),
        ("E1c_indobertweet", "E3_soft_mtl"),
        ("E3_soft_mtl", "E4_consistency"),
        ("E3_soft_mtl", "E5_P1_soft_mtl"),
        ("E3_soft_mtl", "E5_P2_soft_mtl"),
        ("E0_tfidf_lr", "E1c_indobertweet"),
    ]
    rows = []
    for a, b in pairs:
        try:
            da = load_preds(a, args.seed)
            db = load_preds(b, args.seed)
        except FileNotFoundError as e:
            print("[skip pair]", a, b, e)
            continue
        for task in ("tox", "pol"):
            thr_a = threshold_from_metrics(a, args.seed, task)
            thr_b = threshold_from_metrics(b, args.seed, task)
            stats = paired_bootstrap_delta(da, db, task, thr_a, thr_b, n_boot=args.n_boot, seed=args.seed)
            rows.append({"model_a": a, "model_b": b, "task": task, "pred_seed": args.seed, **stats})
            print(a, "->", b, task, stats)

    out_dir = ensure_dir(ROOT / "artifacts" / "metrics" / "bootstrap")
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f"paired_bootstrap_seed{args.seed}.csv", index=False)
    out.to_json(out_dir / f"paired_bootstrap_seed{args.seed}.json", orient="records", indent=2)
    print("Wrote", out_dir / f"paired_bootstrap_seed{args.seed}.csv")


if __name__ == "__main__":
    main()
