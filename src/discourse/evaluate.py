from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

from discourse.calibration import reliability_diagram
from discourse.utils import ensure_dir


@torch.no_grad()
def evaluate_loader(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in tqdm(loader, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()
        for i in range(len(batch["text_id"])):
            rows.append(
                {
                    "text_id": batch["text_id"][i],
                    "prob_tox": float(probs[i, 0]),
                    "prob_pol": float(probs[i, 1]),
                    "n_tox": batch["n_tox"][i],
                    "n_pol": batch["n_pol"][i],
                    "q_tox": batch["q_tox"][i],
                    "q_pol": batch["q_pol"][i],
                    "hard_tox": batch["hard_tox"][i],
                    "hard_pol": batch["hard_pol"][i],
                    "mask_tox": float(batch["target_masks"][i, 0]),
                    "mask_pol": float(batch["target_masks"][i, 1]),
                }
            )
    return pd.DataFrame(rows)


def classification_metrics(pred_df: pd.DataFrame, task: str, threshold: float = 0.5) -> dict:
    hard_col = f"hard_{task}"
    prob_col = f"prob_{task}"
    sub = pred_df[pred_df[hard_col].notna()].copy()
    if len(sub) == 0:
        return {
            "n": 0,
            "macro_f1": float("nan"),
            "pos_f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "average_precision": float("nan"),
            "roc_auc": float("nan"),
        }
    y_true = sub[hard_col].astype(int).to_numpy()
    y_prob = sub[prob_col].to_numpy()
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "n": int(len(sub)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "pos_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1
        else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1
        else float("nan"),
        "threshold": threshold,
    }
    return out


def select_thresholds(pred_df: pd.DataFrame) -> dict[str, float]:
    thresholds = {}
    for task in ("tox", "pol"):
        hard_col = f"hard_{task}"
        prob_col = f"prob_{task}"
        sub = pred_df[pred_df[hard_col].notna()]
        if len(sub) == 0 or sub[hard_col].nunique() < 2:
            thresholds[task] = 0.5
            continue
        y_true = sub[hard_col].astype(int).to_numpy()
        y_prob = sub[prob_col].to_numpy()
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.1, 0.9, 17):
            f1 = f1_score(y_true, (y_prob >= t).astype(int), average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[task] = best_t
    return thresholds


def soft_distribution_metrics(pred_df: pd.DataFrame, task: str) -> dict:
    q_col = f"q_{task}"
    n_col = f"n_{task}"
    prob_col = f"prob_{task}"
    sub = pred_df[(pred_df[n_col] >= 2) & pred_df[q_col].notna()].copy()
    if len(sub) == 0:
        return {"n_multi": 0, "mse": float("nan"), "brier_vs_soft": float("nan")}
    q = sub[q_col].astype(float).to_numpy()
    p = sub[prob_col].astype(float).to_numpy()
    mse = float(np.mean((p - q) ** 2))
    # binary CE against soft target
    eps = 1e-7
    p_clip = np.clip(p, eps, 1 - eps)
    ce = float(np.mean(-(q * np.log(p_clip) + (1 - q) * np.log(1 - p_clip))))
    return {"n_multi": int(len(sub)), "mse": mse, "cross_entropy": ce}


def plot_confusion(pred_df, task, threshold, out_path: Path, title: str):
    hard_col = f"hard_{task}"
    prob_col = f"prob_{task}"
    sub = pred_df[pred_df[hard_col].notna()]
    y_true = sub[hard_col].astype(int)
    y_pred = (sub[prob_col] >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=[0, 1], yticklabels=[0, 1])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cm


def plot_pr_curve(pred_df, task, out_path: Path, title: str):
    hard_col = f"hard_{task}"
    prob_col = f"prob_{task}"
    sub = pred_df[pred_df[hard_col].notna()]
    y_true = sub[hard_col].astype(int).to_numpy()
    y_prob = sub[prob_col].to_numpy()
    if len(np.unique(y_true)) < 2:
        return
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_metrics_and_figures(
    experiment_id: str,
    pred_df: pd.DataFrame,
    thresholds: dict[str, float],
    split_name: str,
    metrics_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)
    rows = []
    for task in ("tox", "pol"):
        m = classification_metrics(pred_df, task, thresholds[task])
        s = soft_distribution_metrics(pred_df, task)
        rows.append(
            {
                "experiment_id": experiment_id,
                "split": split_name,
                "task": task,
                **m,
                **s,
            }
        )
        plot_confusion(
            pred_df,
            task,
            thresholds[task],
            figures_dir / f"{experiment_id}_{split_name}_{task}_cm.png",
            f"G12 {experiment_id} {split_name} {task}",
        )
        plot_pr_curve(
            pred_df,
            task,
            figures_dir / f"{experiment_id}_{split_name}_{task}_pr.png",
            f"G11 {experiment_id} {split_name} {task}",
        )
        # Diagram keandalan disimpan per eksperimen agar setara antar-seed.
        sub = pred_df[pred_df[f"hard_{task}"].notna()]
        if len(sub) and sub[f"hard_{task}"].nunique() > 1:
            reliability_diagram(
                sub[f"hard_{task}"].astype(int).to_numpy(),
                sub[f"prob_{task}"].to_numpy(),
                figures_dir / f"{experiment_id}_{split_name}_{task}_reliability.png",
                f"Reliability {experiment_id} {split_name} {task}",
            )
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(metrics_dir / f"{experiment_id}_{split_name}_metrics.csv", index=False)
    pred_path = metrics_dir.parent / "predictions" / f"{experiment_id}_{split_name}_preds.csv"
    ensure_dir(pred_path.parent)
    pred_df.to_csv(pred_path, index=False)
    return metrics_df
