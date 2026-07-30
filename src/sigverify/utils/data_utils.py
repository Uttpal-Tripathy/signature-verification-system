"""Shared DataLoader sizing so training scripts don't silently produce zero batches
on small datasets (e.g. the synthetic demo set, or a small first pilot collection).
"""
from __future__ import annotations


def safe_batch_size_and_drop_last(dataset_len: int, configured_batch_size: int) -> tuple[int, bool]:
    """Clamp batch size to the dataset size, and only drop the remainder batch when
    at least one full batch would still be left afterwards.
    """
    if dataset_len == 0:
        return configured_batch_size, False
    batch_size = min(configured_batch_size, dataset_len)
    drop_last = dataset_len >= 2 * batch_size
    return batch_size, drop_last
