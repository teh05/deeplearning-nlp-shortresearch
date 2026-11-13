from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class TextCNN(nn.Module):
    def __init__(self, vocab_size, dim=128, channels=128, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.convs = nn.ModuleList(
            [nn.Conv1d(dim, channels, k, padding=k // 2) for k in (3, 5, 7)]
        )
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(channels * 3, 2)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids).transpose(1, 2)
        valid = attention_mask.bool().unsqueeze(1)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x))
            h = h.masked_fill(~valid, torch.finfo(h.dtype).min)
            pooled.append(h.max(dim=-1).values)
        return self.head(self.dropout(torch.cat(pooled, dim=-1)))


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, dim=128, hidden=128, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden * 2, 2))

    def forward(self, input_ids, attention_mask):
        lengths = attention_mask.sum(dim=1).cpu().clamp(min=1)
        x = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(
            packed_h, batch_first=True, total_length=input_ids.size(1)
        )
        scores = self.attention(h).squeeze(-1)
        scores = scores.masked_fill(~attention_mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        z = (h * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(z)


class TransformerMultiTask(nn.Module):
    def __init__(self, checkpoint, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(self.dropout(out.last_hidden_state[:, 0]))
