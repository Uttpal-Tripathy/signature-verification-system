#!/usr/bin/env python
"""Train the static Siamese CNN branch on triplets (anchor/positive genuine, negative
forged-or-impostor). Validation uses a writer-disjoint split and reports EER/AUC —
the standard signature-verification benchmark metrics — every epoch, and checkpoints
the best-EER model.

Usage:
    python scripts/train_static.py --manifest data/processed/demo/static_manifest.jsonl \
        --config configs/default.yaml --output checkpoints/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sigverify.data.datasets import StaticSignatureTripletDataset, split_writers
from sigverify.models.backbones import freeze, unfreeze
from sigverify.models.losses import CombinedEmbeddingLoss
from sigverify.models.static_branch import SiameseCNN
from sigverify.utils.config import load_config
from sigverify.utils.data_utils import safe_batch_size_and_drop_last
from sigverify.utils.logging import get_logger
from sigverify.utils.metrics import verification_report
from sigverify.utils.seed import get_device, set_seed

logger = get_logger(__name__)


@torch.no_grad()
def evaluate(model: SiameseCNN, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    genuine_scores, forgery_scores = [], []
    for anchor, positive, negative in loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        e_a, e_p = model(anchor, positive)
        e_n = model.embed(negative)
        genuine_scores.append(model.similarity(e_a, e_p).cpu().numpy())
        forgery_scores.append(model.similarity(e_a, e_n).cpu().numpy())
    genuine = (np.concatenate(genuine_scores) + 1) / 2  # map cosine [-1,1] -> [0,1] for EER
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
    train_ds = StaticSignatureTripletDataset(args.manifest, target_size=tuple(cfg.preprocessing.image.target_size), writer_ids=train_writers)
    val_ds = StaticSignatureTripletDataset(args.manifest, target_size=tuple(cfg.preprocessing.image.target_size), writer_ids=val_writers)
    train_batch_size, train_drop_last = safe_batch_size_and_drop_last(len(train_ds), cfg.training.batch_size)
    val_batch_size, _ = safe_batch_size_and_drop_last(len(val_ds), cfg.training.batch_size)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=0, drop_last=train_drop_last)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0)
    logger.info("Train triplets=%d (writers=%d) | Val triplets=%d (writers=%d)", len(train_ds), len(train_writers), len(val_ds), len(val_writers))

    model = SiameseCNN(
        backbone=cfg.static_branch.backbone,
        embedding_dim=cfg.static_branch.embedding_dim,
        pretrained=cfg.static_branch.pretrained,
    ).to(device)

    freeze_epochs = cfg.static_branch.freeze_backbone_epochs
    if freeze_epochs > 0:
        freeze(model.extractor)
        logger.info("Backbone frozen for the first %d epoch(s) (embedding head trains alone)", freeze_epochs)

    criterion = CombinedEmbeddingLoss(cfg.static_branch.contrastive_margin, cfg.static_branch.triplet_margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_eer = float("inf")
    patience_left = cfg.training.early_stopping_patience

    for epoch in range(1, epochs + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            unfreeze(model.extractor)
            logger.info("Backbone unfrozen at epoch %d — fine-tuning end-to-end", epoch)

        model.train()
        running_loss = 0.0
        for anchor, positive, negative in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            e_a = model.embed(anchor)
            e_p = model.embed(positive)
            e_n = model.embed(negative)
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
            torch.save(model.state_dict(), output_dir / "static_branch.pt")
            logger.info("New best EER=%.4f — checkpoint saved", best_eer)
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping at epoch %d (best_eer=%.4f)", epoch, best_eer)
                break

    logger.info("Training complete. Best val EER=%.4f", best_eer)


if __name__ == "__main__":
    main()
