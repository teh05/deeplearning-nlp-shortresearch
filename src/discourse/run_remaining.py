# -*- coding: utf-8 -*-
"""Run remaining experiments: E2 STL, E4 consistency, E5 P1/P2 (multi-seed)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from discourse.calibration import brier_score, expected_calibration_error  # noqa: E402
from discourse.evaluate import (  # noqa: E402
    classification_metrics,
    export_metrics_and_figures,
    select_thresholds,
    soft_distribution_metrics,
)
from discourse.perturbation import perturb_text  # noqa: E402
from discourse.predict import load_checkpoint, predict_dataframe  # noqa: E402
from discourse.preprocessing import apply_preprocessing  # noqa: E402
from discourse.train import run_training  # noqa: E402
from discourse.utils import ensure_dir, load_yaml, set_seed  # noqa: E402

SEEDS = [42, 43, 44]


def load_slang() -> dict:
    path = ROOT / "data" / "resources" / "slang_conservative.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def seed_dirs(exp_id: str, seed: int) -> dict[str, Path]:
    base = ROOT / "artifacts"
    return {
        "ckpt": ensure_dir(base / "checkpoints" / exp_id / f"seed_{seed}"),
        "metrics": ensure_dir(base / "metrics" / "seeded"),
        "preds": ensure_dir(base / "predictions" / "seeded"),
        "figures": ensure_dir(base / "figures" / "seeded" / f"{exp_id}_seed{seed}"),
    }


def metrics_from_pred(exp_id, seed, pred_df, thresholds, train_minutes, peak_vram, extra=None):
    rows = []
    for task in ("tox", "pol"):
        m = classification_metrics(pred_df, task, thresholds[task])
        s = soft_distribution_metrics(pred_df, task)
        hard_col = f"hard_{task}"
        prob_col = f"prob_{task}"
        sub = pred_df[pred_df[hard_col].notna()]
        if len(sub):
            y_true = sub[hard_col].astype(int).to_numpy()
            y_prob = sub[prob_col].to_numpy()
            ece = expected_calibration_error(y_true, y_prob)
            brier = brier_score(y_true, y_prob)
        else:
            ece, brier = float("nan"), float("nan")
        row = {
            "experiment_id": exp_id,
            "seed": seed,
            "split": "test",
            "task": task,
            "n": m["n"],
            "threshold": thresholds[task],
            "macro_f1": m["macro_f1"],
            "pos_f1": m["pos_f1"],
            "average_precision": m["average_precision"],
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "roc_auc": m.get("roc_auc"),
            "mse": s.get("mse"),
            "cross_entropy": s.get("cross_entropy"),
            "n_multi": s.get("n_multi"),
            "ece": ece,
            "brier": brier,
            "train_minutes": train_minutes,
            "peak_vram_mb": peak_vram,
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return pd.DataFrame(rows)


def filter_task_rows(df: pd.DataFrame, task_mode: str) -> pd.DataFrame:
    if task_mode == "tox":
        return df[df["mask_tox"].astype(bool) & df["hard_tox"].notna()].copy()
    if task_mode == "pol":
        return df[df["mask_pol"].astype(bool) & df["hard_pol"].notna()].copy()
    return df.copy()


def train_eval_one(
    exp_id: str,
    cfg_path: Path,
    train_df,
    val_df,
    test_df,
    seed: int,
    force: bool = False,
    extra_meta: dict | None = None,
):
    dirs = seed_dirs(exp_id, seed)
    metrics_path = dirs["metrics"] / f"{exp_id}_seed{seed}_test_metrics.csv"
    ckpt_path = dirs["ckpt"] / "best.pt"
    if metrics_path.exists() and ckpt_path.exists() and not force:
        print(f"[skip] {exp_id} seed={seed}")
        return pd.read_csv(metrics_path)

    cfg = load_yaml(cfg_path)
    cfg["seed"] = seed
    cfg["experiment_id"] = exp_id
    task_mode = cfg["training"].get("task_mode", "multitask")
    tr = filter_task_rows(train_df, task_mode)
    va = filter_task_rows(val_df, task_mode)
    print(f"\n=== Training {exp_id} seed={seed} task_mode={task_mode} n_train={len(tr)} ===")
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    result = run_training(cfg, tr, va, out_dir=dirs["ckpt"], device_prefer="cuda")
    elapsed = (time.time() - t0) / 60.0
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None

    model, cfg_loaded, thresholds, vocab, tokenizer, device = load_checkpoint(result["best_path"], "cuda")
    # threshold on full val (for multitask) or task-filtered val
    val_pred = predict_dataframe(model, va if task_mode != "multitask" else val_df, cfg_loaded, vocab, tokenizer, device)
    thresholds = select_thresholds(val_pred)
    test_pred = predict_dataframe(model, test_df, cfg_loaded, vocab, tokenizer, device)
    test_pred["experiment_id"] = exp_id
    test_pred["seed"] = seed
    test_pred["split"] = "test"
    test_pred.to_csv(dirs["preds"] / f"{exp_id}_seed{seed}_test_preds.csv", index=False)
    export_metrics_and_figures(
        f"{exp_id}_seed{seed}",
        test_pred,
        thresholds,
        split_name="test",
        metrics_dir=dirs["metrics"],
        figures_dir=dirs["figures"],
    )
    mdf = metrics_from_pred(exp_id, seed, test_pred, thresholds, elapsed, peak_vram, extra=extra_meta)
    # For STL, keep only the trained task row as primary; still store both for diagnostics
    if task_mode in {"tox", "pol"}:
        mdf = mdf[mdf["task"] == task_mode].copy()
    mdf.to_csv(metrics_path, index=False)
    meta = {
        "experiment_id": exp_id,
        "seed": seed,
        "train_minutes": elapsed,
        "peak_vram_mb": peak_vram,
        "thresholds": thresholds,
        "best_val_mean_macro_f1": result.get("best_score"),
        "checkpoint": str(result["best_path"]),
        "task_mode": task_mode,
        **(extra_meta or {}),
    }
    (dirs["ckpt"] / "seed_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] {exp_id} seed={seed} minutes={elapsed:.2f} peak_vram={peak_vram}")
    return mdf


def merge_e2_stl(seed: int) -> pd.DataFrame:
    """Combine tox/pol STL into E2_stl experiment_id."""
    metrics_dir = ROOT / "artifacts" / "metrics" / "seeded"
    tox = pd.read_csv(metrics_dir / f"E2_stl_tox_seed{seed}_test_metrics.csv")
    pol = pd.read_csv(metrics_dir / f"E2_stl_pol_seed{seed}_test_metrics.csv")
    tox["experiment_id"] = "E2_stl"
    pol["experiment_id"] = "E2_stl"
    merged = pd.concat([tox, pol], ignore_index=True)
    # merge predictions
    preds_dir = ROOT / "artifacts" / "predictions" / "seeded"
    pt = pd.read_csv(preds_dir / f"E2_stl_tox_seed{seed}_test_preds.csv")
    pp = pd.read_csv(preds_dir / f"E2_stl_pol_seed{seed}_test_preds.csv")
    # take tox probs from tox model, pol from pol model (join on text_id)
    pol_cols = ["text_id", "prob_pol"]
    if "logit_pol" in pp.columns:
        pol_cols.append("logit_pol")
    out = pt.drop(columns=[c for c in ["prob_pol", "logit_pol"] if c in pt.columns], errors="ignore")
    out = out.merge(pp[pol_cols], on="text_id", how="left")
    out["experiment_id"] = "E2_stl"
    out.to_csv(preds_dir / f"E2_stl_seed{seed}_test_preds.csv", index=False)
    # recompute thresholds independently already in each; store combined metrics
    path = metrics_dir / f"E2_stl_seed{seed}_test_metrics.csv"
    merged.to_csv(path, index=False)
    return merged


def build_pipeline_df(base_df: pd.DataFrame, pipeline: str, slang: dict | None) -> pd.DataFrame:
    # rebuild text_proc from text_raw with requested pipeline; keep labels/splits
    out = apply_preprocessing(base_df, pipeline=pipeline, slang=slang, text_col="text_raw")
    return out


def eval_perturbed(exp_id: str, seed: int, test_df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Evaluate a checkpoint on mildly perturbed test text."""
    dirs = seed_dirs(exp_id, seed)
    out_path = dirs["metrics"] / f"{exp_id}_seed{seed}_perturbed_test_metrics.csv"
    if out_path.exists() and not force:
        print(f"[skip] perturbed eval {exp_id} seed={seed}")
        return pd.read_csv(out_path)
    ckpt = dirs["ckpt"] / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    model, cfg_loaded, thresholds, vocab, tokenizer, device = load_checkpoint(ckpt, "cuda")
    # rebuild thresholds on clean val not available here; reuse checkpoint thresholds then refine on clean test? Use ckpt thresholds.
    set_seed(seed)
    pert = test_df.copy()
    import random as py_random

    pert["text_proc"] = [
        perturb_text(str(t), rng=py_random.Random(seed + i), mode="mild")
        for i, t in enumerate(pert["text_raw"].tolist())
    ]
    pert["pipeline"] = "P0_perturbed_mild"
    pred = predict_dataframe(model, pert, cfg_loaded, vocab, tokenizer, device)
    pred["experiment_id"] = exp_id
    pred["seed"] = seed
    pred["split"] = "test_perturbed"
    pred.to_csv(dirs["preds"] / f"{exp_id}_seed{seed}_perturbed_test_preds.csv", index=False)
    # compare clean vs perturbed using same thresholds from clean val stored in ckpt
    mdf = metrics_from_pred(
        exp_id,
        seed,
        pred,
        thresholds,
        train_minutes=None,
        peak_vram=None,
        extra={"eval_mode": "perturbed_mild"},
    )
    # attach deltas vs clean if available
    clean_path = dirs["metrics"] / f"{exp_id}_seed{seed}_test_metrics.csv"
    if clean_path.exists():
        clean = pd.read_csv(clean_path)
        rows = []
        for _, r in mdf.iterrows():
            c = clean[(clean.task == r.task)]
            delta = float(r.macro_f1) - float(c.iloc[0].macro_f1) if len(c) else float("nan")
            rr = r.to_dict()
            rr["macro_f1_delta_vs_clean"] = delta
            rows.append(rr)
        mdf = pd.DataFrame(rows)
    mdf.to_csv(out_path, index=False)
    print(f"[done] perturbed eval {exp_id} seed={seed}")
    return mdf


def write_preprocessing_audit(base_df: pd.DataFrame, slang: dict, n: int = 120):
    """Export sample audit rows for P1/P2 meaning checks (human review)."""
    ensure_dir(ROOT / "artifacts" / "audits")
    sample = base_df.sample(n=min(n, len(base_df)), random_state=42)
    rows = []
    for _, r in sample.iterrows():
        raw = str(r["text_raw"])
        for pipe in ("P1", "P2"):
            from discourse.preprocessing import preprocess

            transformed = preprocess(raw, pipeline=pipe, slang=slang if pipe == "P2" else None)
            rows.append(
                {
                    "text_id": r["text_id"],
                    "pipeline": pipe,
                    "original_text": raw[:500],
                    "transformed_text": transformed[:500],
                    "meaning_preserved": "",
                    "toxicity_cue_changed": "",
                    "polarization_cue_changed": "",
                    "reviewer_id": "",
                    "notes": "auto-exported; fill by human",
                }
            )
    out = ROOT / "artifacts" / "audits" / "preprocessing_p1_p2_audit_sample.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Wrote", out)


def write_perturbation_audit(base_df: pd.DataFrame, n: int = 120):
    ensure_dir(ROOT / "artifacts" / "audits")
    sample = base_df.sample(n=min(n, len(base_df)), random_state=42)
    rows = []
    for i, (_, r) in enumerate(sample.iterrows()):
        raw = str(r["text_raw"])
        pert = perturb_text(raw, rng=__import__("random").Random(42 + i), mode="mild")
        rows.append(
            {
                "text_id": r["text_id"],
                "perturbation_type": "mild_ws_url_mention_char_dup",
                "original_text": raw[:500],
                "perturbed_text": pert[:500],
                "meaning_preserved": "",
                "toxicity_cue_changed": "",
                "polarization_cue_changed": "",
                "reviewer_id": "",
                "notes": "auto-exported; fill by human",
            }
        )
    out = ROOT / "artifacts" / "audits" / "perturbation_mild_audit_sample.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Wrote", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--exps",
        nargs="+",
        default=["E2", "E5", "E4"],
        help="Which blocks to run: E2 E5 E4",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    slang = load_slang()
    all_df = pd.read_parquet(ROOT / "data" / "processed" / "all_splits.parquet")
    write_preprocessing_audit(all_df, slang)
    write_perturbation_audit(all_df)

    # cache pipeline variants
    pipe_frames = {"P0": all_df}
    for pipe in ("P1", "P2"):
        path = ROOT / "data" / "processed" / f"all_splits_{pipe}.parquet"
        if path.exists() and not args.force:
            pipe_frames[pipe] = pd.read_parquet(path)
        else:
            print(f"Building {pipe} text_proc ...")
            dfp = build_pipeline_df(all_df, pipe, slang if pipe == "P2" else None)
            dfp.to_parquet(path, index=False)
            pipe_frames[pipe] = dfp

    frames = []

    if "E2" in args.exps:
        for seed in args.seeds:
            for task, cfg_name, exp_id in [
                ("tox", "e2_stl_tox.yaml", "E2_stl_tox"),
                ("pol", "e2_stl_pol.yaml", "E2_stl_pol"),
            ]:
                df = pipe_frames["P0"]
                train_df = df[df.split == "train"]
                val_df = df[df.split == "validation"]
                test_df = df[df.split == "test"]
                mdf = train_eval_one(
                    exp_id,
                    ROOT / "configs" / cfg_name,
                    train_df,
                    val_df,
                    test_df,
                    seed,
                    force=args.force,
                    extra_meta={"pipeline": "P0", "stl_task": task},
                )
                frames.append(mdf)
            frames.append(merge_e2_stl(seed))

    if "E5" in args.exps:
        for pipe in ("P1", "P2"):
            df = pipe_frames[pipe]
            train_df = df[df.split == "train"]
            val_df = df[df.split == "validation"]
            test_df = df[df.split == "test"]
            for seed in args.seeds:
                exp_id = f"E5_{pipe}_soft_mtl"
                mdf = train_eval_one(
                    exp_id,
                    ROOT / "configs" / "e5_soft_mtl.yaml",
                    train_df,
                    val_df,
                    test_df,
                    seed,
                    force=args.force,
                    extra_meta={"pipeline": pipe},
                )
                frames.append(mdf)

    if "E4" in args.exps:
        df = pipe_frames["P0"]
        train_df = df[df.split == "train"]
        val_df = df[df.split == "validation"]
        test_df = df[df.split == "test"]
        for seed in args.seeds:
            mdf = train_eval_one(
                "E4_consistency",
                ROOT / "configs" / "e4_consistency.yaml",
                train_df,
                val_df,
                test_df,
                seed,
                force=args.force,
                extra_meta={"pipeline": "P0", "consistency_weight": 0.5},
            )
            frames.append(mdf)
            frames.append(eval_perturbed("E4_consistency", seed, test_df, force=args.force))
            # also evaluate E3 as clean baseline on same perturbed set if checkpoint exists
            e3_ckpt = ROOT / "artifacts" / "checkpoints" / "E3_soft_mtl" / f"seed_{seed}" / "best.pt"
            if e3_ckpt.exists():
                # symlink-like eval under E3 id
                try:
                    frames.append(eval_perturbed("E3_soft_mtl", seed, test_df, force=args.force))
                except Exception as e:
                    print("[warn] E3 perturbed eval failed:", e)

    # refresh combined seeded metrics
    seeded = ROOT / "artifacts" / "metrics" / "seeded"
    files = list(seeded.glob("*_seed*_test_metrics.csv"))
    if files:
        combined = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
        combined = combined.drop_duplicates(subset=["experiment_id", "seed", "task"], keep="last")
        combined.to_csv(seeded / "all_seeds_test_metrics.csv", index=False)
        print("Wrote", seeded / "all_seeds_test_metrics.csv", "rows=", len(combined))


if __name__ == "__main__":
    main()
