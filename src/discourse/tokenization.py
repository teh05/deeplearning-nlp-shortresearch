from __future__ import annotations

from collections import Counter
from typing import Iterable

from transformers import AutoTokenizer


SPECIALS = ["<pad>", "<unk>"]


def build_word_vocab(texts: Iterable[str], min_freq: int = 2) -> dict[str, int]:
    counter = Counter()
    for t in texts:
        counter.update(str(t).lower().split())
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for tok, freq in counter.items():
        if freq >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def encode_words(text: str, vocab: dict[str, int], max_length: int = 128):
    unk = vocab["<unk>"]
    pad = vocab["<pad>"]
    ids = [vocab.get(tok, unk) for tok in str(text).lower().split()][:max_length]
    if not ids:
        ids = [unk]
    attn = [1] * len(ids)
    while len(ids) < max_length:
        ids.append(pad)
        attn.append(0)
    return ids, attn


def load_hf_tokenizer(checkpoint: str = "indolem/indobertweet-base-uncased"):
    return AutoTokenizer.from_pretrained(checkpoint)
