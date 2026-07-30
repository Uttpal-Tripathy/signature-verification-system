"""SHAP-based modality-contribution attribution for the final decision.

Rather than SHAP-explaining the raw CNN/Transformer pixel-/timestep-space (expensive
and, for a verification task, less actionable than "what drove the accept/reject
call"), this explains the small decision-fusion function itself: it takes the handful
of scalar signals the pipeline actually decides on (fused similarity, static-only
similarity, dynamic-only similarity, anomaly score) and attributes the final
calibrated decision score to each one. That directly answers "how much did the static
image vs. the dynamic stroke data vs. the anomaly check drive this decision" — the
per-modality contribution split the architecture calls for.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import shap

FEATURE_NAMES = ("fused_similarity", "static_similarity", "dynamic_similarity", "anomaly_score")


class DecisionSHAPExplainer:
    def __init__(self, decision_fn: Callable[[np.ndarray], np.ndarray], background: np.ndarray) -> None:
        """decision_fn: (N, 4) array of the FEATURE_NAMES columns -> (N,) calibrated decision score.
        background: (M, 4) representative sample of past decisions, used as the SHAP baseline.
        """
        self.decision_fn = decision_fn
        self.explainer = shap.KernelExplainer(decision_fn, background)

    def explain(self, sample: np.ndarray) -> dict:
        """sample: (4,) array matching FEATURE_NAMES. Returns per-feature SHAP contribution
        toward this specific decision, normalized to a static-vs-dynamic-vs-anomaly split.
        """
        shap_values = self.explainer.shap_values(sample.reshape(1, -1), silent=True)
        values = np.asarray(shap_values).reshape(-1)
        contributions = dict(zip(FEATURE_NAMES, values.tolist()))

        modality_split = self._modality_split(contributions)
        return {"feature_contributions": contributions, "modality_split": modality_split, "base_value": float(self.explainer.expected_value if np.isscalar(self.explainer.expected_value) else self.explainer.expected_value[0])}

    @staticmethod
    def _modality_split(contributions: dict) -> dict:
        static_abs = abs(contributions["static_similarity"])
        dynamic_abs = abs(contributions["dynamic_similarity"])
        anomaly_abs = abs(contributions["anomaly_score"])
        total = static_abs + dynamic_abs + anomaly_abs + 1e-8
        return {
            "static_contribution_pct": round(100 * static_abs / total, 2),
            "dynamic_contribution_pct": round(100 * dynamic_abs / total, 2),
            "anomaly_contribution_pct": round(100 * anomaly_abs / total, 2),
        }
