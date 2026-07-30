"""Metric-learning losses for the static and dynamic branches.

`CombinedEmbeddingLoss` mixes contrastive (pairwise) and triplet (relative) signal,
which the signature-verification literature (e.g. SigNet, TA-SigNet ablations) shows
converges faster and generalizes better to unseen writers than either loss alone —
contrastive gives a hard absolute distance target, triplet teaches relative ordering
between genuine/forged pairs of the *same* writer.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ContrastiveLoss(nn.Module):
    """label=1 for genuine pair (same writer, should be close), label=0 for forgery/impostor pair."""

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embedding_a: torch.Tensor, embedding_b: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        distance = F.pairwise_distance(embedding_a, embedding_b)
        positive_term = label * distance.pow(2)
        negative_term = (1 - label) * F.relu(self.margin - distance).pow(2)
        return (positive_term + negative_term).mean()


class TripletLoss(nn.Module):
    """anchor/positive = same writer genuine signatures, negative = forgery or a different writer."""

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        d_pos = F.pairwise_distance(anchor, positive)
        d_neg = F.pairwise_distance(anchor, negative)
        return F.relu(d_pos - d_neg + self.margin).mean()


class CombinedEmbeddingLoss(nn.Module):
    def __init__(self, contrastive_margin: float = 1.0, triplet_margin: float = 0.3, triplet_weight: float = 0.5) -> None:
        super().__init__()
        self.contrastive = ContrastiveLoss(contrastive_margin)
        self.triplet = TripletLoss(triplet_margin)
        self.triplet_weight = triplet_weight

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        pair_label_pos = torch.ones(anchor.size(0), device=anchor.device)
        pair_label_neg = torch.zeros(anchor.size(0), device=anchor.device)
        contrastive_loss = self.contrastive(anchor, positive, pair_label_pos) + self.contrastive(
            anchor, negative, pair_label_neg
        )
        triplet_loss = self.triplet(anchor, positive, negative)
        return contrastive_loss + self.triplet_weight * triplet_loss
