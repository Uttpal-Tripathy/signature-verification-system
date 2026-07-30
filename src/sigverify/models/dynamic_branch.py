"""Dynamic (online) stroke-sequence branch — LSTM / GRU / Transformer encoder.

Consumes the fixed-length (T, F) stroke tensor produced by
`sigverify.preprocessing.stroke_preprocess` (x, y, pressure, velocity, tilt_x, tilt_y,
timestamp_delta) and produces an L2-normalized embedding comparable to the static
branch's. An attention-pooling head (additive attention for RNNs, a learned query for
the Transformer) both aggregates the sequence AND exposes per-timestep attention
weights, which the explainability module renders as stroke-level deviation heatmaps.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1024) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class AdditiveAttentionPool(nn.Module):
    """Bahdanau-style additive attention that pools a (B, T, D) sequence into (B, D)
    while returning the per-timestep weights for explainability.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(sequence).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)  # (B, D)
        return pooled, weights


class QueryAttentionPool(nn.Module):
    """A single learned query attends over the Transformer's token sequence — the
    Transformer analogue of AdditiveAttentionPool, using nn.MultiheadAttention so the
    returned weights are genuine self-attention scores.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = sequence.size(0)
        query = self.query.expand(batch, -1, -1)
        pooled, weights = self.attn(query, sequence, sequence, need_weights=True, average_attn_weights=True)
        return pooled.squeeze(1), weights.squeeze(1)  # weights: (B, T)


class DynamicStrokeEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        embedding_dim: int = 256,
        encoder: str = "transformer",
        bidirectional: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if encoder == "transformer":
            self.pos_encoding = PositionalEncoding(hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            self.sequence_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.pool = QueryAttentionPool(hidden_dim, num_heads)
            pooled_dim = hidden_dim
        elif encoder in ("lstm", "gru"):
            rnn_cls = nn.LSTM if encoder == "lstm" else nn.GRU
            self.sequence_encoder = rnn_cls(
                hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0,
            )
            pooled_dim = hidden_dim * (2 if bidirectional else 1)
            self.pool = AdditiveAttentionPool(pooled_dim)
        else:
            raise ValueError(f"Unknown dynamic encoder: {encoder}")

        self.output_proj = nn.Sequential(nn.LayerNorm(pooled_dim), nn.Linear(pooled_dim, embedding_dim))

    def forward(self, stroke_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """stroke_seq: (B, T, input_dim) -> (embedding (B, D), attention_weights (B, T))"""
        x = self.input_proj(stroke_seq)
        if self.encoder_type == "transformer":
            x = self.pos_encoding(x)
            x = self.sequence_encoder(x)
        else:
            x, _ = self.sequence_encoder(x)
        pooled, attn_weights = self.pool(x)
        embedding = self.output_proj(pooled)
        return F.normalize(embedding, p=2, dim=1), attn_weights

    @staticmethod
    def similarity(embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> torch.Tensor:
        return (embedding_a * embedding_b).sum(dim=1)
