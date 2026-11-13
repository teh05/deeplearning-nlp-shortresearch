from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from discourse.datasets import DiscourseDataset, collate_batch
from discourse.evaluate import evaluate_loader
from discourse.models import BiLSTMAttention, TextCNN, TransformerMultiTask
from discourse.tokenization import load_hf_tokenizer
from discourse.utils import get_device


def load_checkpoint(path: str | Path, device_prefer: str = "cuda"):
    device = get_device(device_prefer)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    vocab = ckpt.get("vocab")
    name = cfg["model"]["name"]
    if name == "textcnn":
        model = TextCNN(len(vocab), dim=cfg["model"].get("dim", 128), channels=cfg["model"].get("channels", 128))
    elif name == "bilstm":
        model = BiLSTMAttention(len(vocab), dim=cfg["model"].get("dim", 128), hidden=cfg["model"].get("hidden", 128))
    elif name == "transformer":
        model = TransformerMultiTask(cfg["model"]["checkpoint"])
    else:
        raise ValueError(name)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    tokenizer = load_hf_tokenizer(cfg["model"]["checkpoint"]) if name == "transformer" else None
    return model, cfg, ckpt.get("thresholds", {"tox": 0.5, "pol": 0.5}), vocab, tokenizer, device


def predict_dataframe(model, df, cfg, vocab, tokenizer, device, batch_size: int | None = None) -> pd.DataFrame:
    batch_size = batch_size or cfg["training"]["micro_batch_size"]
    ds = DiscourseDataset(
        df,
        tokenizer=tokenizer,
        vocab=vocab,
        max_length=cfg["training"]["max_length"],
        target_mode=cfg["training"].get("target_mode", "hard"),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return evaluate_loader(model, loader, device)
