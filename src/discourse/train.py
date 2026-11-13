from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from discourse.datasets import DiscourseDataset, collate_batch
from discourse.evaluate import evaluate_loader, select_thresholds
from discourse.losses import total_loss
from discourse.models import BiLSTMAttention, TextCNN, TransformerMultiTask
from discourse.tokenization import build_word_vocab, load_hf_tokenizer
from discourse.utils import ensure_dir, get_device, save_json, set_seed


def build_model(cfg: dict, vocab_size: int | None = None):
    name = cfg["model"]["name"]
    if name == "textcnn":
        return TextCNN(
            vocab_size,
            dim=cfg["model"].get("dim", 128),
            channels=cfg["model"].get("channels", 128),
        )
    if name == "bilstm":
        return BiLSTMAttention(
            vocab_size,
            dim=cfg["model"].get("dim", 128),
            hidden=cfg["model"].get("hidden", 128),
        )
    if name == "transformer":
        return TransformerMultiTask(cfg["model"]["checkpoint"])
    raise ValueError(f"Unknown model: {name}")


def train_epoch(model, loader, optimizer, device, accumulation=1, consistency_weight=0.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    n_batches = len(loader)
    tensor_keys = {
        "input_ids",
        "attention_mask",
        "targets",
        "target_masks",
        "aug_input_ids",
        "aug_attention_mask",
    }
    for step, batch in enumerate(loader):
        tensor_batch = {k: v.to(device) for k, v in batch.items() if k in tensor_keys}
        logits = model(tensor_batch["input_ids"], tensor_batch["attention_mask"])
        aug_logits = None
        if consistency_weight > 0 and "aug_input_ids" in tensor_batch:
            aug_logits = model(tensor_batch["aug_input_ids"], tensor_batch["aug_attention_mask"])
        loss = total_loss(
            logits,
            tensor_batch["targets"],
            tensor_batch["target_masks"],
            augmented_logits=aug_logits,
            consistency_weight=consistency_weight,
        )
        window_start = (step // accumulation) * accumulation
        window_size = min(accumulation, n_batches - window_start)
        (loss / window_size).backward()
        if (step + 1) % accumulation == 0 or step + 1 == n_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item()
    return running_loss / max(n_batches, 1)


def run_training(
    cfg: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    out_dir: str | Path,
    device_prefer: str = "cuda",
) -> dict[str, Any]:
    set_seed(cfg.get("seed", 42))
    device = get_device(device_prefer)
    out_dir = ensure_dir(out_dir)
    max_length = cfg["training"]["max_length"]
    target_mode = cfg["training"].get("target_mode", "hard")
    batch_size = cfg["training"]["micro_batch_size"]
    accum = cfg["training"].get("gradient_accumulation", 1)

    tokenizer = None
    vocab = None
    vocab_size = None
    if cfg["model"]["name"] == "transformer":
        tokenizer = load_hf_tokenizer(cfg["model"]["checkpoint"])
    else:
        vocab = build_word_vocab(train_df["text_proc"].tolist())
        vocab_size = len(vocab)
        save_json(vocab, out_dir / "vocab.json")

    task_mode = cfg["training"].get("task_mode", "multitask")
    consistency_weight = float(cfg["training"].get("consistency_weight", 0.0))
    augment_fn = None
    if consistency_weight > 0:
        from discourse.perturbation import make_augment_fn

        augment_fn = make_augment_fn(seed=int(cfg.get("seed", 42)))

    train_ds = DiscourseDataset(
        train_df,
        tokenizer=tokenizer,
        vocab=vocab,
        max_length=max_length,
        target_mode=target_mode,
        task_mode=task_mode,
        augment_fn=augment_fn,
    )
    val_ds = DiscourseDataset(
        val_df,
        tokenizer=tokenizer,
        vocab=vocab,
        max_length=max_length,
        target_mode=target_mode,
        task_mode=task_mode,
        augment_fn=None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    model = build_model(cfg, vocab_size=vocab_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )

    history = []
    best_score = -1.0
    best_path = out_dir / "best.pt"
    thresholds = {"tox": 0.5, "pol": 0.5}

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        tr_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            accumulation=accum,
            consistency_weight=cfg["training"].get("consistency_weight", 0.0),
        )
        val_pred = evaluate_loader(model, val_loader, device)
        thresholds = select_thresholds(val_pred)
        metrics = summarize_preds(val_pred, thresholds, task_mode=task_mode)
        row = {"epoch": epoch, "train_loss": tr_loss, **metrics}
        history.append(row)
        score = metrics.get("mean_macro_f1", 0.0)
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg,
                    "thresholds": thresholds,
                    "vocab": vocab,
                    "epoch": epoch,
                    "metrics": metrics,
                },
                best_path,
            )
        print(
            f"[{cfg.get('experiment_id')}] epoch={epoch} loss={tr_loss:.4f} "
            f"mean_macro_f1={score:.4f} device={device}"
        )

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "history.csv", index=False)
    save_json(
        {
            "experiment_id": cfg.get("experiment_id"),
            "device": str(device),
            "best_score": best_score,
            "thresholds": thresholds,
            "best_checkpoint": str(best_path),
        },
        out_dir / "run_meta.json",
    )
    return {
        "history": hist_df,
        "best_path": best_path,
        "thresholds": thresholds,
        "device": str(device),
        "best_score": best_score,
    }


def summarize_preds(
    pred_df: pd.DataFrame,
    thresholds: dict[str, float],
    task_mode: str = "multitask",
) -> dict[str, float]:
    from discourse.evaluate import classification_metrics

    if task_mode == "tox":
        tasks = [("tox", "tox")]
    elif task_mode == "pol":
        tasks = [("pol", "pol")]
    else:
        tasks = [("tox", "tox"), ("pol", "pol")]

    scores = []
    out = {}
    for task, thr_key in tasks:
        m = classification_metrics(pred_df, task=task, threshold=thresholds[thr_key])
        out[f"{task}_macro_f1"] = m["macro_f1"]
        out[f"{task}_pos_f1"] = m["pos_f1"]
        out[f"{task}_ap"] = m["average_precision"]
        if m["macro_f1"] == m["macro_f1"]:  # not nan
            scores.append(m["macro_f1"])
    out["mean_macro_f1"] = float(sum(scores) / len(scores)) if scores else 0.0
    return out
