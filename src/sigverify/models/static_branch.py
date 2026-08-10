"""Static Siamese CNN branch — learns a visual similarity embedding for signature images.

Architecture follows the SigNet-style Siamese design validated on CEDAR/GPDS/ICDAR
benchmarks: a shared-weight CNN backbone (ImageNet-pretrained for fast convergence on
limited signature data) projected to a compact L2-normalized embedding, trained with a
combined contrastive + triplet objective so both pairwise and relative-distance signal
shape the embedding space.

Two embedding heads are available. `EmbeddingHead` (the original) global-average-pools
the CNN's spatial feature map, discarding *where* in the signature each activation came
from. `HybridEmbeddingHead` instead treats each spatial location as a token and runs a
small Transformer encoder over them before pooling, so the model can attend across
distant regions of the signature (e.g. a stroke's start vs. its flourish at the end) —
the CNN+Transformer hybrid pattern used by recent offline signature verification work
(HTCSigNet, TransOSV, SignatureGuard — see docs/research_gap.md).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sigverify.models.backbones import build_backbone
from sigverify.models.dynamic_branch import QueryAttentionPool


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


class HybridEmbeddingHead(nn.Module):
    """CNN feature map -> spatial tokens -> Transformer encoder (global self-attention
    across regions) -> learned-query attention pooling -> projected embedding.

    `max_tokens` must be >= the backbone's H'*W' at the configured input resolution
    (e.g. mobilenet_v3_large at 128px yields a 4x4=16 token map); the positional
    embedding is sliced to the actual token count so smaller feature maps still work.
    """

    def __init__(
        self,
        in_channels: int,
        embedding_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_tokens: int = 256,
    ) -> None:
        super().__init__()
        self.token_proj = nn.Linear(in_channels, embedding_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_tokens, embedding_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=embedding_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.token_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = QueryAttentionPool(embedding_dim, num_heads)
        self.output_proj = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, embedding_dim))
        self.max_tokens = max_tokens

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        _batch, _channels, height, width = feature_map.shape
        num_tokens = height * width
        if num_tokens > self.max_tokens:
            raise ValueError(f"HybridEmbeddingHead: {num_tokens} spatial tokens exceeds max_tokens={self.max_tokens}; increase max_tokens or lower input resolution.")
        tokens = feature_map.flatten(2).transpose(1, 2)  # (B, H*W, C)
        tokens = self.token_proj(tokens) + self.pos_embedding[:, :num_tokens]
        tokens = self.token_encoder(tokens)
        pooled, _ = self.pool(tokens)
        embedding = self.output_proj(pooled)
        return F.normalize(embedding, p=2, dim=1)


class SiameseCNN(nn.Module):
    """Shared-weight twin network. Call `.embed(x)` for a single image, or `.forward(a, b)`
    for a pair — both go through the identical backbone + head (true weight sharing).
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        embedding_dim: int = 256,
        pretrained: bool = True,
        head_type: str = "cnn",
        head_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.extractor, out_channels = build_backbone(backbone, pretrained=pretrained)
        head_kwargs = head_kwargs or {}
        if head_type == "cnn":
            self.head = EmbeddingHead(out_channels, embedding_dim)
        elif head_type == "hybrid":
            self.head = HybridEmbeddingHead(out_channels, embedding_dim, **head_kwargs)
        else:
            raise ValueError(f"Unknown head_type: {head_type} (expected 'cnn' or 'hybrid')")
        self.backbone_name = backbone
        self.head_type = head_type

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
