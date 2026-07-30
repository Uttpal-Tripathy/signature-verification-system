"""Stroke-level deviation analysis for the dynamic branch.

Combines two signals into one per-timestep deviation score:

1. DTW alignment cost between the query and reference stroke-feature sequences —
   classic online-signature-verification technique for pinpointing *where* two
   sequences diverge, robust to the writer simply signing at a different overall speed.
2. The dynamic encoder's own attention weights (from `DynamicStrokeEncoder`) — which
   timesteps it actually weighted heavily when forming the verification decision.

A stroke that is both poorly aligned with the reference AND heavily attended is the
strongest candidate for "this is the part of the signature that looks forged."
"""
from __future__ import annotations

import numpy as np


def dtw_align(seq_a: np.ndarray, seq_b: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Basic O(N*M) DTW. seq_a, seq_b: (T, F). Returns (cost_matrix, alignment_path)."""
    n, m = len(seq_a), len(seq_b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    dist = np.linalg.norm(seq_a[:, None, :] - seq_b[None, :, :], axis=-1)  # (n, m)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = dist[i - 1, j - 1] + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return cost[1:, 1:], path


def stroke_deviation_scores(
    query_seq: np.ndarray,
    reference_seq: np.ndarray,
    query_attention_weights: np.ndarray,
    attention_weight_mix: float = 0.5,
) -> np.ndarray:
    """Returns a (len(query_seq),) deviation score in [0, 1], one per query timestep,
    blending normalized DTW misalignment cost with the encoder's attention weight.
    """
    _, path = dtw_align(query_seq, reference_seq)
    point_dist = np.array([np.linalg.norm(query_seq[i] - reference_seq[j]) for i, j in path])

    # Multiple reference points can align to one query point (or vice versa); average them.
    per_query_dist = np.zeros(len(query_seq))
    counts = np.zeros(len(query_seq))
    for (i, _j), d in zip(path, point_dist):
        per_query_dist[i] += d
        counts[i] += 1
    counts[counts == 0] = 1
    per_query_dist /= counts

    dist_norm = (per_query_dist - per_query_dist.min()) / (per_query_dist.max() - per_query_dist.min() + 1e-8)
    attn = np.asarray(query_attention_weights)
    attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    return attention_weight_mix * attn_norm + (1 - attention_weight_mix) * dist_norm


def top_k_deviant_indices(deviation_scores: np.ndarray, k: int = 5) -> np.ndarray:
    return np.argsort(deviation_scores)[::-1][:k]
