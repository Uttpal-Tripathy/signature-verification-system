#!/usr/bin/env python
"""Train the CycleGAN forgery synthesizer on unpaired genuine/forged static images,
then use the trained genuine->forged generator to synthesize additional skilled
forgeries and append them to the manifest for the static branch's next training run
— the augmentation half of the closed adversarial loop (Gap C). Mining the static
branch's actual false negatives/positives back into `FailureCaseBuffer` for targeted
retraining happens at evaluation time (see scripts/evaluate.py), once a trained
verifier exists to generate those failure cases from.

Usage:
    python scripts/train_gan.py --manifest data/processed/demo/static_manifest.jsonl \
        --config configs/default.yaml --output checkpoints/ --synthesize 200
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sigverify.data.datasets import load_manifest
from sigverify.models.gan_forgery import CycleGANForgerySynthesizer
from sigverify.preprocessing.image_preprocess import preprocess_signature_image
from sigverify.utils.config import load_config
from sigverify.utils.data_utils import safe_batch_size_and_drop_last
from sigverify.utils.logging import get_logger
from sigverify.utils.seed import get_device, set_seed

logger = get_logger(__name__)


class UnpairedDomainDataset(Dataset):
    """Randomly pairs one genuine and one forged image per index — CycleGAN needs no
    correspondence between the two domains, only enough samples of each.
    """

    def __init__(self, manifest_path: str, target_size: tuple[int, int]):
        records = load_manifest(manifest_path)
        self.genuine = [r["path"] for r in records if r["label"] == "genuine"]
        self.forged = [r["path"] for r in records if r["label"] == "forged"]
        self.target_size = target_size
        self.length = max(len(self.genuine), len(self.forged))
        if not self.genuine or not self.forged:
            raise ValueError("Manifest must contain both genuine and forged samples to train the GAN")

    def __len__(self) -> int:
        return self.length

    def _load(self, path: str) -> torch.Tensor:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        processed = preprocess_signature_image(raw, target_size=self.target_size)
        return torch.from_numpy(processed).unsqueeze(0) * 2 - 1  # map [0,1] -> [-1,1] to match generator's Tanh output

    def __getitem__(self, idx: int):
        genuine_path = self.genuine[idx % len(self.genuine)]
        forged_path = random.choice(self.forged)
        return self._load(genuine_path), self._load(forged_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--synthesize", type=int, default=0, help="Number of extra synthetic forgeries to append to the manifest after training")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    epochs = args.epochs or cfg.training.epochs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = UnpairedDomainDataset(args.manifest, tuple(cfg.preprocessing.image.target_size))
    batch_size, drop_last = safe_batch_size_and_drop_last(len(dataset), cfg.gan_augmentation.batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    gan = CycleGANForgerySynthesizer(
        in_channels=1,
        generator_channels=cfg.gan_augmentation.generator_channels,
        discriminator_channels=cfg.gan_augmentation.discriminator_channels,
        num_residual_blocks=cfg.gan_augmentation.num_residual_blocks,
        lambda_cycle=cfg.gan_augmentation.lambda_cycle,
        lambda_identity=cfg.gan_augmentation.lambda_identity,
    ).to(device)

    gen_params = list(gan.genuine_to_forged.parameters()) + list(gan.forged_to_genuine.parameters())
    disc_params = list(gan.discriminator_forged.parameters()) + list(gan.discriminator_genuine.parameters())
    gen_optimizer = torch.optim.Adam(gen_params, lr=cfg.training.lr, betas=(0.5, 0.999))
    disc_optimizer = torch.optim.Adam(disc_params, lr=cfg.training.lr, betas=(0.5, 0.999))

    for epoch in range(1, epochs + 1):
        gan.train()
        running = {"gen": 0.0, "disc": 0.0}
        for real_genuine, real_forged in tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            real_genuine, real_forged = real_genuine.to(device), real_forged.to(device)

            gen_optimizer.zero_grad()
            gen_out = gan.generator_loss(real_genuine, real_forged)
            gen_out["total"].backward()
            gen_optimizer.step()

            disc_optimizer.zero_grad()
            d_forged = gan.discriminator_loss(real_forged, gen_out["fake_forged"], gan.discriminator_forged)
            d_genuine = gan.discriminator_loss(real_genuine, gen_out["fake_genuine"], gan.discriminator_genuine)
            (d_forged + d_genuine).backward()
            disc_optimizer.step()

            running["gen"] += gen_out["total"].item()
            running["disc"] += (d_forged + d_genuine).item()

        n = max(1, len(loader))
        logger.info("epoch %d | gen_loss=%.4f | disc_loss=%.4f", epoch, running["gen"] / n, running["disc"] / n)

    torch.save(gan.state_dict(), output_dir / "gan_forgery.pt")
    logger.info("Saved GAN checkpoint to %s", output_dir / "gan_forgery.pt")

    if args.synthesize > 0:
        _synthesize_and_append(gan, dataset, args.manifest, args.synthesize, device, tuple(cfg.preprocessing.image.target_size))


def _synthesize_and_append(gan, dataset, manifest_path: str, count: int, device, target_size) -> None:
    manifest_path = Path(manifest_path)
    synth_dir = manifest_path.parent / "gan_synthetic_forgeries"
    synth_dir.mkdir(parents=True, exist_ok=True)

    gan.eval()
    records = load_manifest(manifest_path)
    genuine_records = [r for r in records if r["label"] == "genuine"]

    new_records = []
    for i in range(count):
        rec = random.choice(genuine_records)
        raw = cv2.imread(rec["path"], cv2.IMREAD_UNCHANGED)
        processed = preprocess_signature_image(raw, target_size=target_size)
        tensor = (torch.from_numpy(processed).unsqueeze(0).unsqueeze(0) * 2 - 1).to(device)
        fake_forged = gan.generate_forgery(tensor).squeeze().cpu().numpy()
        fake_forged = ((fake_forged + 1) / 2 * 255).astype(np.uint8)

        out_path = synth_dir / f"gan_{i:04d}.png"
        cv2.imwrite(str(out_path), fake_forged)
        new_records.append({"path": str(out_path), "writer_id": rec["writer_id"], "label": "forged", "forgery_type": "gan_synthetic"})

    with open(manifest_path, "a", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in new_records)
    logger.info("Appended %d GAN-synthesized forgeries to %s", len(new_records), manifest_path)


if __name__ == "__main__":
    main()
