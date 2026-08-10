#!/usr/bin/env python
"""Writer-disjoint K-fold cross-validation for the static or dynamic branch.

A single train/val split (what train_static.py / train_dynamic.py report) has real
variance at these writer counts: which handful of writers land in the held-out set
measurably moves EER/AUC on its own, independent of anything about the model. K-fold
cross-validation holds out every writer exactly once, across K folds, and reports the
mean +/- standard deviation of the per-fold metrics — the standard way to get a
defensible number out of a small-writer-count benchmark, and normal practice for a
paper or patent submission where a single lucky/unlucky split isn't citable.

This script also doubles as the harness for comparing two architecture variants
head-to-head on identical folds: run it once with the baseline head/encoder and once
with the hybrid CNN-Transformer (static) / BiLSTM-Transformer (dynamic) variant, same
--seed, same --folds, same --max-writers, and the two runs are directly comparable
because every fold sees exactly the same train/val writer split.

Usage:
    python scripts/cross_validate.py --branch static \
        --manifest data/processed/real/cedar_manifest.jsonl --config configs/lightweight_real.yaml \
        --folds 5 --max-writers 25 --head-type cnn --output cv_results/static_cnn

    python scripts/cross_validate.py --branch dynamic \
        --manifest data/processed/real/mobisig_manifest.jsonl --config configs/lightweight_real.yaml \
        --folds 5 --max-writers 20 --encoder hybrid --output cv_results/dynamic_hybrid
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sigverify.data.datasets import (
    DynamicStrokeTripletDataset,
    StaticSignatureTripletDataset,
    kfold_writers,
)
from sigverify.models.backbones import freeze, unfreeze
from sigverify.models.dynamic_branch import DynamicStrokeEncoder
from sigverify.models.losses import CombinedEmbeddingLoss
from sigverify.models.static_branch import SiameseCNN
from sigverify.utils.config import load_config
from sigverify.utils.data_utils import safe_batch_size_and_drop_last
from sigverify.utils.logging import get_logger
from sigverify.utils.metrics import evaluation_matrix, verification_report
from sigverify.utils.plotting import plot_confusion_matrix, plot_roc_curve
from sigverify.utils.seed import get_device, set_seed

logger = get_logger(__name__)


@torch.no_grad()
def _evaluate_static(model: SiameseCNN, loader: DataLoader, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    genuine_scores, forgery_scores = [], []
    for anchor, positive, negative in loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        e_a, e_p = model(anchor, positive)
        e_n = model.embed(negative)
        genuine_scores.append(model.similarity(e_a, e_p).cpu().numpy())
        forgery_scores.append(model.similarity(e_a, e_n).cpu().numpy())
    genuine = (np.concatenate(genuine_scores) + 1) / 2
    forgery = (np.concatenate(forgery_scores) + 1) / 2
    return verification_report(genuine, forgery), genuine, forgery


@torch.no_grad()
def _evaluate_dynamic(model: DynamicStrokeEncoder, loader: DataLoader, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray]:
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
    return verification_report(genuine, forgery), genuine, forgery


def _run_static_fold(args, cfg, device, fold_idx, train_writers, val_writers, shared_cache):
    target_size = tuple(cfg.preprocessing.image.target_size)
    train_ds = StaticSignatureTripletDataset(
        args.manifest, target_size=target_size, writer_ids=train_writers,
        cache_in_memory=True, augment=args.augment, shared_denoise_cache=shared_cache,
    )
    val_ds = StaticSignatureTripletDataset(
        args.manifest, target_size=target_size, writer_ids=val_writers,
        cache_in_memory=True, shared_denoise_cache=shared_cache,
    )
    train_batch_size, train_drop_last = safe_batch_size_and_drop_last(len(train_ds), cfg.training.batch_size)
    val_batch_size, _ = safe_batch_size_and_drop_last(len(val_ds), cfg.training.batch_size)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=0, drop_last=train_drop_last)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0)

    model = SiameseCNN(
        backbone=cfg.static_branch.backbone, embedding_dim=cfg.static_branch.embedding_dim,
        pretrained=cfg.static_branch.pretrained, head_type=args.head_type,
    ).to(device)
    freeze_epochs = cfg.static_branch.freeze_backbone_epochs
    if freeze_epochs > 0:
        freeze(model.extractor)
    criterion = CombinedEmbeddingLoss(cfg.static_branch.contrastive_margin, cfg.static_branch.triplet_margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    epochs = args.epochs or cfg.training.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_eer, best_metrics, best_scores = float("inf"), None, None
    patience_left = cfg.training.early_stopping_patience
    for epoch in range(1, epochs + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            unfreeze(model.extractor)
        model.train()
        running_loss = 0.0
        for anchor, positive, negative in tqdm(train_loader, desc=f"fold {fold_idx} epoch {epoch}/{epochs}", leave=False):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            e_a, e_p, e_n = model.embed(anchor), model.embed(positive), model.embed(negative)
            loss = criterion(e_a, e_p, e_n)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        metrics, val_genuine, val_forgery = _evaluate_static(model, val_loader, device)
        logger.info(
            "fold %d epoch %d | train_loss=%.4f | val_eer=%.4f | val_auc=%.4f | val_acc@eer=%.4f",
            fold_idx, epoch, running_loss / max(1, len(train_loader)), metrics["eer"], metrics["roc_auc"], metrics["accuracy_at_eer_threshold"],
        )
        if metrics["eer"] < best_eer:
            best_eer, best_metrics = metrics["eer"], metrics
            best_scores = (val_genuine, val_forgery)
            patience_left = cfg.training.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("fold %d early stopping at epoch %d (best_eer=%.4f)", fold_idx, epoch, best_eer)
                break

    return best_metrics, best_scores, len(train_writers), len(val_writers)


def _run_dynamic_fold(args, cfg, device, fold_idx, train_writers, val_writers):
    common_kwargs = {"resample_points": cfg.preprocessing.stroke.resample_points, "normalize": cfg.preprocessing.stroke.normalize}
    train_ds = DynamicStrokeTripletDataset(args.manifest, writer_ids=train_writers, **common_kwargs)
    val_ds = DynamicStrokeTripletDataset(args.manifest, writer_ids=val_writers, **common_kwargs)
    train_batch_size, train_drop_last = safe_batch_size_and_drop_last(len(train_ds), cfg.training.batch_size)
    val_batch_size, _ = safe_batch_size_and_drop_last(len(val_ds), cfg.training.batch_size)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=0, drop_last=train_drop_last)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0)

    model = DynamicStrokeEncoder(
        input_dim=cfg.dynamic_branch.input_dim, hidden_dim=cfg.dynamic_branch.hidden_dim,
        num_layers=cfg.dynamic_branch.num_layers, num_heads=cfg.dynamic_branch.num_heads,
        embedding_dim=cfg.dynamic_branch.embedding_dim, encoder=args.encoder or cfg.dynamic_branch.encoder,
        bidirectional=cfg.dynamic_branch.bidirectional, dropout=cfg.dynamic_branch.dropout,
    ).to(device)
    criterion = CombinedEmbeddingLoss(cfg.static_branch.contrastive_margin, cfg.static_branch.triplet_margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    epochs = args.epochs or cfg.training.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_eer, best_metrics, best_scores = float("inf"), None, None
    patience_left = cfg.training.early_stopping_patience
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for anchor, positive, negative in tqdm(train_loader, desc=f"fold {fold_idx} epoch {epoch}/{epochs}", leave=False):
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

        metrics, val_genuine, val_forgery = _evaluate_dynamic(model, val_loader, device)
        logger.info(
            "fold %d epoch %d | train_loss=%.4f | val_eer=%.4f | val_auc=%.4f | val_acc@eer=%.4f",
            fold_idx, epoch, running_loss / max(1, len(train_loader)), metrics["eer"], metrics["roc_auc"], metrics["accuracy_at_eer_threshold"],
        )
        if metrics["eer"] < best_eer:
            best_eer, best_metrics = metrics["eer"], metrics
            best_scores = (val_genuine, val_forgery)
            patience_left = cfg.training.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("fold %d early stopping at epoch %d (best_eer=%.4f)", fold_idx, epoch, best_eer)
                break

    return best_metrics, best_scores, len(train_writers), len(val_writers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--branch", required=True, choices=["static", "dynamic"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/lightweight_real.yaml")
    parser.add_argument("--output", required=True, help="Directory for cv_summary.json and the pooled ROC curve")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None, help="Only actually run the first N of the K folds computed above (K still determines the writer split geometry) — for a quick, still-writer-disjoint, representative confusion matrix without paying for all K folds")
    parser.add_argument("--max-writers", type=int, default=None, help="Cap the writer pool to bound CPU training time; stated explicitly in the saved summary")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true", help="static branch only")
    parser.add_argument("--head-type", default="cnn", choices=["cnn", "hybrid"], help="static branch only")
    parser.add_argument("--encoder", default=None, choices=["transformer", "lstm", "gru", "hybrid"], help="dynamic branch only; overrides config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = kfold_writers(args.manifest, k=args.folds, seed=args.seed, max_writers=args.max_writers)
    if args.max_folds is not None:
        splits = splits[: args.max_folds]
    variant = args.head_type if args.branch == "static" else (args.encoder or cfg.dynamic_branch.encoder)
    num_folds_run = len(splits)
    logger.info(
        "%d-fold CV (of %d total) | branch=%s | variant=%s | writer pool=%d%s",
        num_folds_run, args.folds, args.branch, variant, sum(len(v) for _, v in splits),
        f" (capped from full dataset via --max-writers {args.max_writers})" if args.max_writers else " (full dataset)",
    )

    shared_cache: dict = {} if args.branch == "static" else None
    fold_results = []
    pooled_genuine, pooled_forgery = [], []
    for fold_idx, (train_writers, val_writers) in enumerate(splits, start=1):
        set_seed(cfg.seed + fold_idx)  # distinct init per fold, still fully reproducible
        if args.branch == "static":
            metrics, scores, n_train, n_val = _run_static_fold(args, cfg, device, fold_idx, train_writers, val_writers, shared_cache)
        else:
            metrics, scores, n_train, n_val = _run_dynamic_fold(args, cfg, device, fold_idx, train_writers, val_writers)

        fold_results.append({"fold": fold_idx, "train_writers": n_train, "val_writers": n_val, **metrics})
        pooled_genuine.append(scores[0])
        pooled_forgery.append(scores[1])
        logger.info("fold %d/%d done | eer=%.4f auc=%.4f acc@eer=%.4f", fold_idx, num_folds_run, metrics["eer"], metrics["roc_auc"], metrics["accuracy_at_eer_threshold"])

    eers = np.array([r["eer"] for r in fold_results])
    aucs = np.array([r["roc_auc"] for r in fold_results])
    accs = np.array([r["accuracy_at_eer_threshold"] for r in fold_results])

    pooled_genuine = np.concatenate(pooled_genuine)
    pooled_forgery = np.concatenate(pooled_forgery)
    # Pooled (all folds' held-out predictions concatenated) confusion matrix and full
    # evaluation metrics — every sample here came from a fold where its writer was in
    # the held-out set, so this is still a writer-disjoint evaluation, just reported
    # as one matrix instead of five.
    pooled_eval = evaluation_matrix(pooled_genuine, pooled_forgery)

    summary = {
        "branch": args.branch,
        "variant": variant,
        "folds": num_folds_run,
        "folds_configured": args.folds,
        "max_writers": args.max_writers,
        "manifest": str(args.manifest),
        "config": str(args.config),
        "per_fold": fold_results,
        "mean_eer": float(eers.mean()), "std_eer": float(eers.std()),
        "mean_auc": float(aucs.mean()), "std_auc": float(aucs.std()),
        "mean_accuracy_at_eer": float(accs.mean()), "std_accuracy_at_eer": float(accs.std()),
        "pooled_confusion_matrix": {"tp": pooled_eval["tp"], "fn": pooled_eval["fn"], "fp": pooled_eval["fp"], "tn": pooled_eval["tn"]},
        "pooled_evaluation_matrix": pooled_eval,
    }
    with open(output_dir / "cv_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    np.savez(output_dir / "pooled_scores.npz", genuine=pooled_genuine, forgery=pooled_forgery)

    plot_roc_curve(
        pooled_genuine, pooled_forgery,
        f"{args.branch} branch, {variant} — {num_folds_run}-fold pooled CV ROC (mean EER={eers.mean():.4f}+/-{eers.std():.4f})",
        output_dir / "cv_pooled_roc.png",
    )
    plot_confusion_matrix(
        pooled_genuine, pooled_forgery,
        f"{args.branch} branch, {variant} — {num_folds_run}-fold pooled CV confusion matrix",
        output_dir / "cv_confusion_matrix.png",
    )

    logger.info("=" * 70)
    logger.info("%d-FOLD CV RESULT | branch=%s | variant=%s", num_folds_run, args.branch, variant)
    logger.info("EER          : %.4f +/- %.4f", eers.mean(), eers.std())
    logger.info("ROC-AUC      : %.4f +/- %.4f", aucs.mean(), aucs.std())
    logger.info("Acc @ EER    : %.4f +/- %.4f", accs.mean(), accs.std())
    logger.info("Pooled confusion matrix: TP=%d FN=%d FP=%d TN=%d", pooled_eval["tp"], pooled_eval["fn"], pooled_eval["fp"], pooled_eval["tn"])
    logger.info("Pooled precision=%.4f recall=%.4f specificity=%.4f F1=%.4f", pooled_eval["precision"], pooled_eval["recall"], pooled_eval["specificity"], pooled_eval["f1_score"])
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
