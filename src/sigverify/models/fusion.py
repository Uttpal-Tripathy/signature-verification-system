"""Cross-Attention Fusion Network (Gated Residual + Attention) — Gap A.

Combines the static-image embedding and the dynamic-stroke embedding into one fused
representation. Two ideas from the fusion literature are combined:

1. Cross-attention: the two modality embeddings are treated as a length-2 sequence and
   attend to each other, letting each modality's representation be refined by context
   from the other before they're combined.
2. Reliability gating: a small head estimates a scalar reliability score per modality
   (signatures captured without a stylus have no dynamic data at all; noisy scans hurt
   the static signal) and uses it to softmax-weight the residual combination, so a
   corrupted/absent modality is automatically down-weighted rather than corrupting the
   fused embedding.

Also handles the graceful-degradation case (dynamic modality unavailable, e.g. a
scanned-only signature with no stylus capture) by falling back to the static
embedding alone with a modality mask.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ReliabilityGate(nn.Module):
    """Estimates a scalar reliability score per modality from its own embedding statistics."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score_fn = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(inplace=True), nn.Linear(dim // 2, 1))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.score_fn(embedding).squeeze(-1)  # (B,)


class CrossAttentionGatedFusion(nn.Module):
    def __init__(self, embedding_dim: int = 256, num_heads: int = 4, dropout: float = 0.1, reliability_gating: bool = True) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.reliability_gating = reliability_gating

        self.cross_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        if reliability_gating:
            self.static_gate = ReliabilityGate(embedding_dim)
            self.dynamic_gate = ReliabilityGate(embedding_dim)

        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        static_embedding: torch.Tensor,
        dynamic_embedding: torch.Tensor | None = None,
        modality_mask: torch.Tensor | None = None,
    ) -> dict:
        """Returns dict with fused_embedding (B, D) plus per-modality contribution weights
        (B,) each, so downstream explainability can report the static-vs-dynamic split.

        `modality_mask`: optional (B,) bool tensor, True where dynamic data is available
        for that sample. Samples with no dynamic capture fall back to static-only.
        """
        batch = static_embedding.size(0)
        device = static_embedding.device

        if dynamic_embedding is None:
            modality_mask = torch.zeros(batch, dtype=torch.bool, device=device)
            dynamic_embedding = torch.zeros_like(static_embedding)
        elif modality_mask is None:
            modality_mask = torch.ones(batch, dtype=torch.bool, device=device)

        # Treat (static, dynamic) as a length-2 sequence and let them cross-attend.
        pair = torch.stack([static_embedding, dynamic_embedding], dim=1)  # (B, 2, D)
        attended, _ = self.cross_attn(pair, pair, pair, need_weights=False)
        pair = self.norm1(pair + attended)
        pair = self.norm2(pair + self.ffn(pair))
        static_refined, dynamic_refined = pair[:, 0, :], pair[:, 1, :]

        if self.reliability_gating:
            static_score = self.static_gate(static_refined)
            dynamic_score = self.dynamic_gate(dynamic_refined)
        else:
            static_score = torch.ones(batch, device=device)
            dynamic_score = torch.ones(batch, device=device)

        # Force dynamic weight to zero wherever the modality is absent, then renormalize.
        dynamic_score = dynamic_score.masked_fill(~modality_mask, float("-inf"))
        weights = F.softmax(torch.stack([static_score, dynamic_score], dim=1), dim=1)  # (B, 2)
        static_weight, dynamic_weight = weights[:, 0], weights[:, 1]

        # dynamic_weight is exactly 0 for masked-out samples (softmax of -inf), so the
        # zero-embedding placeholder for missing dynamic data contributes nothing.
        fused = static_weight.unsqueeze(1) * static_refined + dynamic_weight.unsqueeze(1) * dynamic_refined

        fused = self.output_proj(fused)
        fused = F.normalize(fused, p=2, dim=1)

        return {
            "fused_embedding": fused,
            "static_weight": static_weight.detach(),
            "dynamic_weight": dynamic_weight.detach(),
        }

    @staticmethod
    def similarity(embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> torch.Tensor:
        return (embedding_a * embedding_b).sum(dim=1)
