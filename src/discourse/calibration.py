from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from discourse.utils import ensure_dir


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if not np.any(m):
            continue
        conf = y_prob[m].mean()
        acc = y_true[m].mean()
        ece += (m.mean()) * abs(acc - conf)
    return float(ece)


def brier_score(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_diagram(y_true, y_prob, out_path: Path, title: str, n_bins: int = 10):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, accs, confs, counts = [], [], [], []
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if not np.any(m):
            continue
        bin_centers.append((bins[i] + bins[i + 1]) / 2)
        accs.append(y_true[m].mean())
        confs.append(y_prob[m].mean())
        counts.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.bar(bin_centers, accs, width=0.08, alpha=0.7, label="accuracy", color="#3b82f6")
    ax.plot(bin_centers, confs, "o-", color="#ef4444", label="confidence")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame({"bin_center": bin_centers, "accuracy": accs, "confidence": confs, "count": counts})


def calibrate_report(pred_df: pd.DataFrame, experiment_id: str, figures_dir: Path, tables_dir: Path):
    rows = []
    for task in ("tox", "pol"):
        hard_col = f"hard_{task}"
        prob_col = f"prob_{task}"
        sub = pred_df[pred_df[hard_col].notna()]
        if len(sub) == 0:
            continue
        y_true = sub[hard_col].astype(int).to_numpy()
        y_prob = sub[prob_col].to_numpy()
        ece = expected_calibration_error(y_true, y_prob)
        brier = brier_score(y_true, y_prob)
        rows.append(
            {
                "experiment_id": experiment_id,
                "task": task,
                "n": len(sub),
                "ece": ece,
                "brier": brier,
            }
        )
        reliability_diagram(
            y_true,
            y_prob,
            Path(figures_dir) / f"{experiment_id}_{task}_reliability.png",
            f"G13 Reliability — {experiment_id} {task}",
        )
    out = pd.DataFrame(rows)
    ensure_dir(tables_dir)
    out.to_csv(Path(tables_dir) / f"{experiment_id}_calibration.csv", index=False)
    return out
