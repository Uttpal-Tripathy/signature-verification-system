"""Grad-CAM adapted for the Siamese static branch.

Standard Grad-CAM needs a classification logit to differentiate; a Siamese network
has none. Instead we differentiate the *pairwise cosine similarity* between the query
and reference embeddings with respect to the query image's final conv feature map —
the resulting heatmap highlights the strokes/regions that most increased (or, for a
mismatch, most decreased) agreement with the reference signature. This is the
established adaptation used in Siamese-network explainability work.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from sigverify.models.static_branch import SiameseCNN


class SiameseGradCAM:
    def __init__(self, model: SiameseCNN) -> None:
        self.model = model

    def explain(self, query_image: torch.Tensor, reference_image: torch.Tensor) -> dict:
        """query_image/reference_image: (1, C, H, W). Returns the heatmap (H, W) in [0, 1]
        resized to the input resolution, plus the raw similarity score.
        """
        self.model.eval()
        query_image = query_image.clone().requires_grad_(True)

        feature_map = self.model.feature_map(query_image)  # (1, C, h, w), needs grad
        feature_map.retain_grad()
        query_embedding = self.model.head(feature_map)

        with torch.no_grad():
            reference_embedding = self.model.embed(reference_image)

        similarity = self.model.similarity(query_embedding, reference_embedding)
        self.model.zero_grad(set_to_none=True)
        similarity.sum().backward()

        gradients = feature_map.grad  # (1, C, h, w)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # global-average-pool the gradients
        cam = F.relu((weights * feature_map).sum(dim=1, keepdim=True))  # (1, 1, h, w)

        cam = F.interpolate(cam, size=query_image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return {"heatmap": cam, "similarity": float(similarity.item())}

    @staticmethod
    def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """image: (H, W) or (H, W, 3) float in [0, 1]. Returns an RGB uint8 overlay."""
        from matplotlib import cm

        if image.ndim == 2:
            base = np.stack([image] * 3, axis=-1)
        else:
            base = image
        colored = cm.jet(heatmap)[..., :3]
        overlay = (1 - alpha) * base + alpha * colored
        return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
