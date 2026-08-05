"""Shared plotting helpers so every training script and notebook renders ROC curves
(and the EER operating point on them) the same way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sigverify.utils.metrics import equal_error_rate, fpr_tpr_curve, roc_auc


def plot_roc_curve(
    genuine_scores: np.ndarray,
    forgery_scores: np.ndarray,
    title: str,
    output_path: str | Path | None = None,
    ax=None,
):
    """Plots the ROC curve (FPR vs. TPR) with the EER operating point marked, and the
    diagonal chance line for reference. Returns the matplotlib Axes; saves to
    `output_path` if given.
    """
    import matplotlib

    if output_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = fpr_tpr_curve(genuine_scores, forgery_scores)
    auc = roc_auc(genuine_scores, forgery_scores)
    eer_stats = equal_error_rate(genuine_scores, forgery_scores)

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(fpr, tpr, color="tab:blue", linewidth=2, label=f"ROC (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.scatter(
        [eer_stats["far"]], [1 - eer_stats["frr"]],
        color="tab:red", zorder=5, s=60,
        label=f"EER={eer_stats['eer']:.4f} @ threshold={eer_stats['threshold']:.3f}",
    )
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (1 - FRR)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    if owns_fig:
        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
    return ax
