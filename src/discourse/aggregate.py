# -*- coding: utf-8 -*-
"""Aggregate per-seed metrics into mean±std tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDED = ROOT / "artifacts" / "metrics" / "seeded"
OUT = ROOT / "artifacts" / "metrics" / "aggregates"

METRIC_COLS = [
    "macro_f1",
    "pos_f1",
    "average_precision",
    "mse",
    "cross_entropy",
    "ece",
    "brier",
    "train_minutes",
    "peak_vram_mb",
]


def main():
    files = sorted(
        p
        for p in SEEDED.glob("*_seed*_test_metrics.csv")
        if "perturbed" not in p.name and p.name != "all_seeds_test_metrics.csv"
    )
    if not files:
        raise SystemExit(f"No seeded metrics in {SEEDED}")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df = df.drop_duplicates(subset=["experiment_id", "seed", "task"], keep="last")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(SEEDED / "all_seeds_test_metrics.csv", index=False)

    # Robustness: clean vs perturbed deltas
    pert_files = sorted(SEEDED.glob("*_seed*_perturbed_test_metrics.csv"))
    if pert_files:
        pert = pd.concat([pd.read_csv(p) for p in pert_files], ignore_index=True)
        pert.to_csv(SEEDED / "all_seeds_perturbed_test_metrics.csv", index=False)
        rob_rows = []
        for (exp, task), g in pert.groupby(["experiment_id", "task"]):
            vals = pd.to_numeric(g.get("macro_f1_delta_vs_clean"), errors="coerce").dropna()
            f1 = pd.to_numeric(g.get("macro_f1"), errors="coerce").dropna()
            rob_rows.append(
                {
                    "experiment_id": exp,
                    "task": task,
                    "n_seeds": int(g["seed"].nunique()),
                    "perturbed_macro_f1_mean": float(f1.mean()) if len(f1) else None,
                    "perturbed_macro_f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
                    "delta_vs_clean_mean": float(vals.mean()) if len(vals) else None,
                    "delta_vs_clean_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                }
            )
        pd.DataFrame(rob_rows).to_csv(OUT / "robustness_perturbed_mean_std.csv", index=False)

    rows = []
    for (exp, task), g in df.groupby(["experiment_id", "task"]):
        seeds = sorted(g["seed"].unique().tolist())
        for metric in METRIC_COLS:
            if metric not in g.columns:
                continue
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            rows.append(
                {
                    "experiment_id": exp,
                    "task": task,
                    "metric": metric,
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "n_seeds": int(len(vals)),
                    "seeds": ";".join(str(s) for s in seeds),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }
            )
    agg = pd.DataFrame(rows)
    agg.to_csv(OUT / "mean_std_by_experiment.csv", index=False)

    # wide summary for report
    pivot_rows = []
    for (exp, task), g in df.groupby(["experiment_id", "task"]):
        row = {"experiment_id": exp, "task": task, "n_seeds": g["seed"].nunique()}
        for metric in ["macro_f1", "pos_f1", "average_precision", "mse", "cross_entropy", "ece", "brier"]:
            if metric not in g.columns:
                continue
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            if len(vals) == 0:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
            else:
                row[f"{metric}_mean"] = float(vals.mean())
                row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        pivot_rows.append(row)
    wide = pd.DataFrame(pivot_rows)
    wide.to_csv(OUT / "summary_mean_std_wide.csv", index=False)

    # paired deltas for key scientific comparisons
    comparisons = [
        ("E2_stl", "E1c_indobertweet", "E1c_minus_E2"),  # MTL hard - STL hard
        ("E1c_indobertweet", "E3_soft_mtl", "E3_minus_E1c"),  # soft - hard MTL
        ("E3_soft_mtl", "E4_consistency", "E4_minus_E3"),
        ("E3_soft_mtl", "E5_P1_soft_mtl", "E5P1_minus_E3"),
        ("E3_soft_mtl", "E5_P2_soft_mtl", "E5P2_minus_E3"),
        ("E0_tfidf_lr", "E1c_indobertweet", "E1c_minus_E0"),
    ]
    delta_rows = []
    for left, right, name in comparisons:
        a_all = df[df.experiment_id == left]
        b_all = df[df.experiment_id == right]
        if a_all.empty or b_all.empty:
            continue
        for task in ["tox", "pol"]:
            a = a_all[a_all.task == task].set_index("seed")
            b = b_all[b_all.task == task].set_index("seed")
            common = sorted(set(a.index) & set(b.index))
            for metric in ["macro_f1", "mse", "cross_entropy", "ece", "brier", "average_precision"]:
                if metric not in a.columns or metric not in b.columns:
                    continue
                diffs = []
                for s in common:
                    va = a.loc[s, metric]
                    vb = b.loc[s, metric]
                    if pd.notna(va) and pd.notna(vb):
                        diffs.append(float(vb) - float(va))
                if not diffs:
                    continue
                ser = pd.Series(diffs)
                delta_rows.append(
                    {
                        "comparison": name,
                        "model_a": left,
                        "model_b": right,
                        "task": task,
                        "metric": metric,
                        "mean_delta": float(ser.mean()),
                        "std_delta": float(ser.std(ddof=1)) if len(ser) > 1 else 0.0,
                        "n_seeds": len(ser),
                        "seeds": ";".join(str(s) for s in common),
                    }
                )
    if delta_rows:
        pd.DataFrame(delta_rows).to_csv(OUT / "pairwise_deltas.csv", index=False)
        # keep legacy filename for E3-E1c
        legacy = [r for r in delta_rows if r["comparison"] == "E3_minus_E1c"]
        if legacy:
            pd.DataFrame(legacy).to_csv(OUT / "pairwise_delta_E3_minus_E1c.csv", index=False)

    print("Aggregated experiments:", sorted(df.experiment_id.unique()))
    print("Seeds:", sorted(df.seed.unique()))
    print("Wrote", OUT / "mean_std_by_experiment.csv")
    print("Wrote", OUT / "summary_mean_std_wide.csv")
    if delta_rows:
        print("Wrote", OUT / "pairwise_deltas.csv")
    print(wide.to_string(index=False))


if __name__ == "__main__":
    main()
