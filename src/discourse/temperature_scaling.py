# -*- coding: utf-8 -*-
"""Fit temperature scaling on validation; report ECE/Brier before/after on test."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from discourse.calibration import brier_score, expected_calibration_error  # noqa: E402
from discourse.datasets import DiscourseDataset, collate_batch  # noqa: E402
from discourse.predict import load_checkpoint  # noqa: E402
from discourse.utils import ensure_dir  # noqa: E402


def collect_logits(model, df, cfg, vocab, tokenizer, device):
    ds = DiscourseDataset(
        df,
        tokenizer=tokenizer,
        vocab=vocab,
        max_length=cfg["training"]["max_length"],
        target_mode=cfg["training"].get("target_mode", "hard"),
    )
    loader = DataLoader(ds, batch_size=cfg["training"]["micro_batch_size"], shuffle=False, collate_fn=collate_batch)
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device)).cpu()
            for i, tid in enumerate(batch["text_id"]):
                rows.append(
                    {
                        "text_id": tid,
                        "logit_tox": float(logits[i, 0]),
                        "logit_pol": float(logits[i, 1]),
                        "hard_tox": batch["hard_tox"][i],
                        "hard_pol": batch["hard_pol"][i],
                    }
                )
    return pd.DataFrame(rows)


def fit_temperature(logits: np.ndarray, y: np.ndarray, masks: np.ndarray) -> float:
    """Single scalar T > 0 minimizing NLL on labeled rows."""
    if masks.sum() == 0:
        return 1.0
    t = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=50)
    logits_t = torch.tensor(logits[masks], dtype=torch.float32)
    y_t = torch.tensor(y[masks], dtype=torch.float32)

    def closure():
        opt.zero_grad()
        temp = t.clamp(min=1e-3)
        loss = F.binary_cross_entropy_with_logits(logits_t / temp, y_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.clamp(min=1e-3).detach().item())


def eval_task(df, task, T=1.0):
    hard = f"hard_{task}"
    logit = f"logit_{task}"
    sub = df[df[hard].notna()].copy()
    y = sub[hard].astype(int).to_numpy()
    p = 1.0 / (1.0 + np.exp(-sub[logit].to_numpy() / T))
    return {
        "ece": float(expected_calibration_error(y, p)),
        "brier": float(brier_score(y, p)),
        "n": int(len(sub)),
        "T": float(T),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exps",
        nargs="+",
        default=["E1c_indobertweet", "E3_soft_mtl", "E4_consistency", "E2_stl_tox", "E2_stl_pol"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_df = pd.read_parquet(ROOT / "data" / "processed" / "all_splits.parquet")
    val_df = all_df[all_df.split == "validation"].copy()
    test_df = all_df[all_df.split == "test"].copy()
    out_dir = ensure_dir(ROOT / "artifacts" / "metrics" / "calibration")
    rows = []

    for exp_id in args.exps:
        ckpt = ROOT / "artifacts" / "checkpoints" / exp_id / f"seed_{args.seed}" / "best.pt"
        if not ckpt.exists():
            print("[skip]", exp_id)
            continue
        print("Fitting T for", exp_id)
        model, cfg, thresholds, vocab, tokenizer, device = load_checkpoint(ckpt, "cuda")
        val_logits = collect_logits(model, val_df, cfg, vocab, tokenizer, device)
        test_logits = collect_logits(model, test_df, cfg, vocab, tokenizer, device)

        temps = {}
        for task in ("tox", "pol"):
            hard = f"hard_{task}"
            logit = f"logit_{task}"
            y = val_logits[hard].to_numpy()
            masks = ~pd.isna(y)
            y_bin = np.where(masks, y.astype(float), 0.0)
            T = fit_temperature(val_logits[logit].to_numpy(), y_bin, masks)
            # for STL single-task models, opposite task may be untrained; still fit
            temps[task] = T
            before = eval_task(test_logits, task, T=1.0)
            after = eval_task(test_logits, task, T=T)
            rows.append(
                {
                    "experiment_id": exp_id,
                    "seed": args.seed,
                    "task": task,
                    "temperature": T,
                    "ece_before": before["ece"],
                    "ece_after": after["ece"],
                    "brier_before": before["brier"],
                    "brier_after": after["brier"],
                    "n_test": after["n"],
                }
            )
            print(exp_id, task, f"T={T:.3f}", "ECE", before["ece"], "->", after["ece"])
        (out_dir / f"{exp_id}_seed{args.seed}_temperature.json").write_text(
            json.dumps(temps, indent=2), encoding="utf-8"
        )

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f"temperature_scaling_seed{args.seed}.csv", index=False)
    print("Wrote", out_dir / f"temperature_scaling_seed{args.seed}.csv")


if __name__ == "__main__":
    main()
