#!/usr/bin/env python
"""Train the cross-attention gated fusion network on top of the (frozen) pretrained
static + dynamic branches. Requires both branches to already be trained
(train_static.py / train_dynamic.py) since the fusion layer learns to combine their
embedding spaces, not to build new ones.

Assumes the static and dynamic manifests were generated together (e.g. by
scripts/generate_demo_data.py) so each writer/label bucket has matching sample counts
— true for any dataset where every stroke capture also has a corresponding scanned
image of the same physical signature.

Usage:
    python scripts/train_fusion.py --static-manifest data/processed/demo/static_manifest.jsonl \
        --dynamic-manifest data/processed/demo/dynamic_manifest.jsonl \
        --checkpoints checkpoints/ --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sigverify.data.datasets import load_manifest, split_writers
from sigverify.models.dynamic_branch import DynamicStrokeEncoder
from sigverify.models.fusion import CrossAttentionGatedFusion
from sigverify.models.losses import CombinedEmbeddingLoss
from sigverify.models.static_branch import SiameseCNN
from sigverify.preprocessing.image_preprocess import preprocess_signature_image
from sigverify.preprocessing.stroke_preprocess import preprocess_stroke_sequence
from sigverify.utils.config import load_config
from sigverify.utils.data_utils import safe_batch_size_and_drop_last
from sigverify.utils.logging import get_logger
from sigverify.utils.metrics import verification_report
from sigverify.utils.seed import get_device, set_seed

logger = get_logger(__name__)


class PairedTripletDataset(Dataset):
    """Anchor/positive/negative triplets where each element is a (static_image,
    stroke_sequence) pair for the same physical signature capture.
    """

    def __init__(self, static_manifest: str, dynamic_manifest: str, cfg, writer_ids: set[str] | None = None, impostor_ratio: float = 0.5):
        self.cfg = cfg
        self.impostor_ratio = impostor_ratio
        static_records = load_manifest(static_manifest)
        dynamic_records = load_manifest(dynamic_manifest)

        def bucket(records):
            out: dict[tuple[str, str], list[dict]] = {}
            for rec in records:
                out.setdefault((rec["writer_id"], rec["label"]), []).append(rec)
            return out

        static_by_key = bucket(static_records)
        dynamic_by_key = bucket(dynamic_records)

        self.paired_by_writer_label: dict[tuple[str, str], list[tuple[dict, dict]]] = {}
        for key, static_list in static_by_key.items():
            dynamic_list = dynamic_by_key.get(key, [])
            if writer_ids is not None and key[0] not in writer_ids:
                continue
            n = min(len(static_list), len(dynamic_list))
            self.paired_by_writer_label[key] = list(zip(static_list[:n], dynamic_list[:n]))

        self.writers = sorted({key[0] for key in self.paired_by_writer_label})
        self.anchors = [pair for key, pairs in self.paired_by_writer_label.items() if key[1] == "genuine" and len(pairs) >= 2 for pair in pairs]

    def __len__(self) -> int:
        return len(self.anchors)

    def _load(self, static_rec: dict, dynamic_rec: dict) -> tuple[torch.Tensor, torch.Tensor]:
        raw = cv2.imread(static_rec["path"], cv2.IMREAD_UNCHANGED)
        image = preprocess_signature_image(raw, target_size=tuple(self.cfg.preprocessing.image.target_size))
        import json

        with open(dynamic_rec["path"], "r", encoding="utf-8") as fh:
            stroke = json.load(fh)
        stroke_matrix = preprocess_stroke_sequence(stroke, self.cfg.preprocessing.stroke.resample_points, self.cfg.preprocessing.stroke.normalize)
        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(stroke_matrix)

    def __getitem__(self, idx: int):
        anchor_static, anchor_dynamic = self.anchors[idx]
        writer = anchor_static["writer_id"]

        genuine_pairs = self.paired_by_writer_label.get((writer, "genuine"), [])
        positive_pool = [p for p in genuine_pairs if p[0]["path"] != anchor_static["path"]]
        positive_static, positive_dynamic = random.choice(positive_pool) if positive_pool else (anchor_static, anchor_dynamic)

        forged_pairs = self.paired_by_writer_label.get((writer, "forged"), [])
        if forged_pairs and random.random() > self.impostor_ratio:
            negative_static, negative_dynamic = random.choice(forged_pairs)
        else:
            other_writers = [w for w in self.writers if w != writer] or [writer]
            other_writer = random.choice(other_writers)
            other_genuine = self.paired_by_writer_label.get((other_writer, "genuine"), genuine_pairs)
            negative_static, negative_dynamic = random.choice(other_genuine)

        return (
            self._load(anchor_static, anchor_dynamic),
            self._load(positive_static, positive_dynamic),
            self._load(negative_static, negative_dynamic),
        )


def collate_paired(batch):
    def stack(items, index):
        return torch.stack([b[index][0] for b in items]), torch.stack([b[index][1] for b in items])

    return stack(batch, 0), stack(batch, 1), stack(batch, 2)


@torch.no_grad()
def embed_pair(static_model, dynamic_model, fusion_model, images, strokes, device):
    images, strokes = images.to(device), strokes.to(device)
    static_emb = static_model.embed(images)
    dynamic_emb, _ = dynamic_model(strokes)
    mask = torch.ones(images.size(0), dtype=torch.bool, device=device)
    return fusion_model(static_emb, dynamic_emb, mask)["fused_embedding"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-manifest", required=True)
    parser.add_argument("--dynamic-manifest", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    epochs = args.epochs or cfg.training.epochs
    ckpt_dir = Path(args.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    static_model = SiameseCNN(cfg.static_branch.backbone, cfg.static_branch.embedding_dim, cfg.static_branch.pretrained).to(device)
    dynamic_model = DynamicStrokeEncoder(
        cfg.dynamic_branch.input_dim, cfg.dynamic_branch.hidden_dim, cfg.dynamic_branch.num_layers,
        cfg.dynamic_branch.num_heads, cfg.dynamic_branch.embedding_dim, cfg.dynamic_branch.encoder,
        cfg.dynamic_branch.bidirectional, cfg.dynamic_branch.dropout,
    ).to(device)
    for path, model in [(ckpt_dir / "static_branch.pt", static_model), (ckpt_dir / "dynamic_branch.pt", dynamic_model)]:
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
            logger.info("Loaded pretrained %s", path)
        else:
            logger.warning("%s not found — fusion will train on randomly-initialized branch embeddings", path)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    fusion_model = CrossAttentionGatedFusion(cfg.fusion.embedding_dim, cfg.fusion.num_heads, cfg.fusion.dropout, cfg.fusion.reliability_gating).to(device)

    train_writers, val_writers = split_writers(args.static_manifest, val_fraction=0.2, seed=cfg.seed)
    train_ds = PairedTripletDataset(args.static_manifest, args.dynamic_manifest, cfg, writer_ids=train_writers)
    val_ds = PairedTripletDataset(args.static_manifest, args.dynamic_manifest, cfg, writer_ids=val_writers)
    train_batch_size, train_drop_last = safe_batch_size_and_drop_last(len(train_ds), cfg.training.batch_size)
    val_batch_size, _ = safe_batch_size_and_drop_last(len(val_ds), cfg.training.batch_size)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, collate_fn=collate_paired, drop_last=train_drop_last)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, collate_fn=collate_paired)
    logger.info("Train triplets=%d | Val triplets=%d", len(train_ds), len(val_ds))

    criterion = CombinedEmbeddingLoss(cfg.static_branch.contrastive_margin, cfg.static_branch.triplet_margin)
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    best_eer = float("inf")
    for epoch in range(1, epochs + 1):
        fusion_model.train()
        running_loss = 0.0
        for anchor, positive, negative in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            (a_img, a_stroke), (p_img, p_stroke), (n_img, n_stroke) = anchor, positive, negative
            with torch.no_grad():
                a_static, a_dynamic = static_model.embed(a_img.to(device)), dynamic_model(a_stroke.to(device))[0]
                p_static, p_dynamic = static_model.embed(p_img.to(device)), dynamic_model(p_stroke.to(device))[0]
                n_static, n_dynamic = static_model.embed(n_img.to(device)), dynamic_model(n_stroke.to(device))[0]
            mask = torch.ones(a_img.size(0), dtype=torch.bool, device=device)

            optimizer.zero_grad()
            e_a = fusion_model(a_static, a_dynamic, mask)["fused_embedding"]
            e_p = fusion_model(p_static, p_dynamic, mask)["fused_embedding"]
            e_n = fusion_model(n_static, n_dynamic, mask)["fused_embedding"]
            loss = criterion(e_a, e_p, e_n)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        fusion_model.eval()
        genuine_scores, forgery_scores = [], []
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                (a_img, a_stroke), (p_img, p_stroke), (n_img, n_stroke) = anchor, positive, negative
                e_a = embed_pair(static_model, dynamic_model, fusion_model, a_img, a_stroke, device)
                e_p = embed_pair(static_model, dynamic_model, fusion_model, p_img, p_stroke, device)
                e_n = embed_pair(static_model, dynamic_model, fusion_model, n_img, n_stroke, device)
                genuine_scores.append(fusion_model.similarity(e_a, e_p).cpu().numpy())
                forgery_scores.append(fusion_model.similarity(e_a, e_n).cpu().numpy())
        genuine = (np.concatenate(genuine_scores) + 1) / 2
        forgery = (np.concatenate(forgery_scores) + 1) / 2
        metrics = verification_report(genuine, forgery)
        logger.info("epoch %d | train_loss=%.4f | val_eer=%.4f | val_auc=%.4f", epoch, running_loss / max(1, len(train_loader)), metrics["eer"], metrics["roc_auc"])

        if metrics["eer"] < best_eer:
            best_eer = metrics["eer"]
            torch.save(fusion_model.state_dict(), ckpt_dir / "fusion.pt")
            logger.info("New best fused EER=%.4f — checkpoint saved", best_eer)

    logger.info("Fusion training complete. Best fused val EER=%.4f", best_eer)


if __name__ == "__main__":
    main()
