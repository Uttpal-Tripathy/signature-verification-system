#!/usr/bin/env python
"""Train the dynamic (online) stroke-sequence branch (LSTM/GRU/Transformer encoder)
on triplets of resampled stroke sequences. Same writer-disjoint EER/AUC evaluation
protocol as train_static.py so the two branches are directly comparable.

Usage:
    python scripts/train_dynamic.py --manifest data/processed/demo/dynamic_manifest.jsonl \
        --config configs/default.yaml --output checkpoints/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sigverify.data.datasets import DynamicStrokeTripletDataset, split_writers
from sigverify.models.dynamic_branch import DynamicStrokeEncoder
from sigverify.models.losses import CombinedEmbeddingLoss
from sigverify.utils.config import load_config
from sigverify.utils.data_utils import safe_batch_size_and_drop_last
from sigverify.utils.logging import get_logger
from sigverify.utils.metrics import verification_report
from sigverify.utils.seed import get_device, set_seed

logger = get_logger(__name__)


@torch.no_grad()
def evaluate(model: DynamicStrokeEncoder, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    genuine_scores, forgery_scores = [], []
    for anchor, positive, negative in loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        e_a, _ = model(anchor)
        e_p, _ = model(positive)
        e_n, _ = model(negative)
        genuine_scores.append(model.similarity(e_a, e_p).cpu().numpy())
        forgery_scores.append(model.similarity(e_a, e_n).cpu().numpy())
    genuine = (np.concatenate(genuine_scores) + 1) / 2
    forgery = (np.concatenate(forgery_scores) + 1) / 2
    return verification_report(genuine, forgery)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    epochs = args.epochs or cfg.training.epochs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_writers, val_writers = split_writers(args.manifest, val_fraction=0.2, seed=cfg.seed)
    common_kwargs = {"resample_points": cfg.preprocessing.stroke.resample_points, "normalize": cfg.preprocessing.stroke.normalize}
    train_ds = DynamicStrokeTripletDataset(args.manifest, writer_ids=train_writers, **common_kwargs)
    val_ds = DynamicStrokeTripletDataset(args.manifest, writer_ids=val_writers, **common_kwargs)
    train_batch_size, train_drop_last = safe_batch_size_and_drop_last(len(train_ds), cfg.training.batch_size)
    val_batch_size, _ = safe_batch_size_and_drop_last(len(val_ds), cfg.training.batch_size)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=0, drop_last=train_drop_last)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0)
    logger.info("Train triplets=%d (writers=%d) | Val triplets=%d (writers=%d)", len(train_ds), len(train_writers), len(val_ds), len(val_writers))

    model = DynamicStrokeEncoder(
        input_dim=cfg.dynamic_branch.input_dim,
        hidden_dim=cfg.dynamic_branch.hidden_dim,
        num_layers=cfg.dynamic_branch.num_layers,
        num_heads=cfg.dynamic_branch.num_heads,
        embedding_dim=cfg.dynamic_branch.embedding_dim,
        encoder=cfg.dynamic_branch.encoder,
        bidirectional=cfg.dynamic_branch.bidirectional,
        dropout=cfg.dynamic_branch.dropout,
    ).to(device)
    criterion = CombinedEmbeddingLoss(cfg.static_branch.contrastive_margin, cfg.static_branch.triplet_margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_eer = float("inf")
    patience_left = cfg.training.early_stopping_patience

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for anchor, positive, negative in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            e_a, _ = model(anchor)
            e_p, _ = model(positive)
            e_n, _ = model(negative)
            loss = criterion(e_a, e_p, e_n)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        metrics = evaluate(model, val_loader, device)
        logger.info(
            "epoch %d | train_loss=%.4f | val_eer=%.4f | val_auc=%.4f | val_acc@eer=%.4f",
            epoch, running_loss / max(1, len(train_loader)), metrics["eer"], metrics["roc_auc"], metrics["accuracy_at_eer_threshold"],
        )

        if metrics["eer"] < best_eer:
            best_eer = metrics["eer"]
            patience_left = cfg.training.early_stopping_patience
            torch.save(model.state_dict(), output_dir / "dynamic_branch.pt")
            logger.info("New best EER=%.4f — checkpoint saved", best_eer)
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping at epoch %d (best_eer=%.4f)", epoch, best_eer)
                break

    logger.info("Training complete. Best val EER=%.4f", best_eer)


if __name__ == "__main__":
    main()
