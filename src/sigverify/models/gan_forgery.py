"""GAN-based skilled-forgery synthesis (CycleGAN) — training-time-only augmentation, Gap C.

Learns a genuine -> forged mapping (and its inverse, for cycle consistency) so the
static branch sees a much larger and more diverse population of *skilled* forgeries
during training than any single dataset provides, directly targeting the
skilled-forgery blind spot most signature-verification papers flag as their main
failure mode.

`FailureCaseBuffer` implements the closed adversarial retraining loop described in the
architecture: verifier false negatives/positives (forgeries that fooled the verifier,
or genuine signatures it rejected) are mined back into the GAN's training set each
retraining cycle, so the generator is continually pushed to produce forgeries that
specifically target the verifier's *current* weaknesses rather than a static
distribution.
"""
from __future__ import annotations

from collections import deque

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CycleGANGenerator(nn.Module):
    """ResNet-style generator (Johnson et al.), the standard CycleGAN backbone."""

    def __init__(self, in_channels: int = 1, base_channels: int = 64, num_residual_blocks: int = 6) -> None:
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, 7),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        ]
        channels = base_channels
        for _ in range(2):  # downsample x2
            layers += [
                nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(channels * 2),
                nn.ReLU(inplace=True),
            ]
            channels *= 2
        layers += [ResidualBlock(channels) for _ in range(num_residual_blocks)]
        for _ in range(2):  # upsample x2
            layers += [
                nn.ConvTranspose2d(channels, channels // 2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(channels // 2),
                nn.ReLU(inplace=True),
            ]
            channels //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(channels, in_channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN — classifies overlapping local patches as real/fake rather than the
    whole image, which sharpens sensitivity to local stroke-texture artifacts.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()

        def block(c_in: int, c_out: int, stride: int = 2, normalize: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(c_in, c_out, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(c_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, base_channels, normalize=False),
            *block(base_channels, base_channels * 2),
            *block(base_channels * 2, base_channels * 4),
            *block(base_channels * 4, base_channels * 8, stride=1),
            nn.Conv2d(base_channels * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CycleGANForgerySynthesizer(nn.Module):
    """Bundles both generators + both discriminators and exposes the three CycleGAN
    losses (adversarial, cycle-consistency, identity) needed to train them.
    """

    def __init__(
        self,
        in_channels: int = 1,
        generator_channels: int = 64,
        discriminator_channels: int = 64,
        num_residual_blocks: int = 6,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 0.5,
    ) -> None:
        super().__init__()
        self.genuine_to_forged = CycleGANGenerator(in_channels, generator_channels, num_residual_blocks)
        self.forged_to_genuine = CycleGANGenerator(in_channels, generator_channels, num_residual_blocks)
        self.discriminator_forged = PatchGANDiscriminator(in_channels, discriminator_channels)
        self.discriminator_genuine = PatchGANDiscriminator(in_channels, discriminator_channels)

        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.adversarial_loss = nn.MSELoss()  # LSGAN formulation: more stable than vanilla BCE
        self.cycle_loss = nn.L1Loss()
        self.identity_loss = nn.L1Loss()

    def generate_forgery(self, genuine_image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.genuine_to_forged(genuine_image)

    def generator_loss(self, real_genuine: torch.Tensor, real_forged: torch.Tensor) -> dict[str, torch.Tensor]:
        fake_forged = self.genuine_to_forged(real_genuine)
        fake_genuine = self.forged_to_genuine(real_forged)
        recon_genuine = self.forged_to_genuine(fake_forged)
        recon_forged = self.genuine_to_forged(fake_genuine)
        identity_genuine = self.forged_to_genuine(real_genuine)
        identity_forged = self.genuine_to_forged(real_forged)

        valid_forged = torch.ones_like(self.discriminator_forged(fake_forged))
        valid_genuine = torch.ones_like(self.discriminator_genuine(fake_genuine))

        adv = self.adversarial_loss(self.discriminator_forged(fake_forged), valid_forged) + self.adversarial_loss(
            self.discriminator_genuine(fake_genuine), valid_genuine
        )
        cycle = self.cycle_loss(recon_genuine, real_genuine) + self.cycle_loss(recon_forged, real_forged)
        identity = self.identity_loss(identity_genuine, real_genuine) + self.identity_loss(identity_forged, real_forged)

        total = adv + self.lambda_cycle * cycle + self.lambda_identity * identity
        return {"total": total, "adversarial": adv, "cycle": cycle, "identity": identity,
                "fake_forged": fake_forged.detach(), "fake_genuine": fake_genuine.detach()}

    def discriminator_loss(self, real: torch.Tensor, fake: torch.Tensor, discriminator: nn.Module) -> torch.Tensor:
        real_pred = discriminator(real)
        fake_pred = discriminator(fake.detach())
        real_loss = self.adversarial_loss(real_pred, torch.ones_like(real_pred))
        fake_loss = self.adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
        return (real_loss + fake_loss) * 0.5


class FailureCaseBuffer:
    """Ring buffer of verifier failure cases (skilled forgeries that scored as genuine,
    or genuine signatures that scored as forged), mined back into GAN retraining so the
    forgery generator keeps chasing the verifier's current decision boundary rather
    than a fixed forgery distribution (the "closed adversarial loop").
    """

    def __init__(self, capacity: int = 2000) -> None:
        self.false_negatives = deque(maxlen=capacity)  # forgeries misclassified as genuine
        self.false_positives = deque(maxlen=capacity)  # genuine misclassified as forged

    def add_false_negative(self, image: torch.Tensor) -> None:
        self.false_negatives.append(image.detach().cpu())

    def add_false_positive(self, image: torch.Tensor) -> None:
        self.false_positives.append(image.detach().cpu())

    def sample_retraining_batch(self, batch_size: int) -> dict[str, torch.Tensor] | None:
        if len(self.false_negatives) < batch_size or len(self.false_positives) < batch_size:
            return None
        import random

        fn_batch = random.sample(list(self.false_negatives), batch_size)
        fp_batch = random.sample(list(self.false_positives), batch_size)
        return {"hard_forgeries": torch.stack(fn_batch), "hard_genuine": torch.stack(fp_batch)}

    def __len__(self) -> int:
        return len(self.false_negatives) + len(self.false_positives)
