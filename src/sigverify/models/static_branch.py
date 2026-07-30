"""Static Siamese CNN branch — learns a visual similarity embedding for signature images.

Architecture follows the SigNet-style Siamese design validated on CEDAR/GPDS/ICDAR
benchmarks: a shared-weight CNN backbone (ImageNet-pretrained for fast convergence on
limited signature data) projected to a compact L2-normalized embedding, trained with a
combined contrastive + triplet objective so both pairwise and relative-distance signal
shape the embedding space.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sigverify.models.backbones import build_backbone


class EmbeddingHead(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int = 256) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.LayerNorm(512),  # unlike BatchNorm1d, well-defined for batch_size==1 (small/last-batch triplet mining)
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(feature_map).flatten(1)
        embedding = self.projector(pooled)
        return F.normalize(embedding, p=2, dim=1)


class SiameseCNN(nn.Module):
    """Shared-weight twin network. Call `.embed(x)` for a single image, or `.forward(a, b)`
    for a pair — both go through the identical backbone + head (true weight sharing).
    """

    def __init__(self, backbone: str = "resnet50", embedding_dim: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        self.extractor, out_channels = build_backbone(backbone, pretrained=pretrained)
        self.head = EmbeddingHead(out_channels, embedding_dim)
        self.backbone_name = backbone

    def _to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Raw conv feature map (pre-pooling) — used directly by the Grad-CAM explainer."""
        return self.extractor(self._to_rgb(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.feature_map(x))

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embed(img_a), self.embed(img_b)

    @staticmethod
    def similarity(embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> torch.Tensor:
        """Cosine similarity in [-1, 1]; embeddings are already L2-normalized so this is a dot product."""
        return (embedding_a * embedding_b).sum(dim=1)
