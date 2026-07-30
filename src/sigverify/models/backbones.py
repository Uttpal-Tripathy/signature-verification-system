"""CNN backbone factory shared by the static Siamese branch and its Grad-CAM explainer."""
from __future__ import annotations

import torchvision.models as tvm
from torch import nn

_BACKBONE_OUT_DIMS = {
    "resnet50": 2048,
    "efficientnet_b0": 1280,
    "mobilenet_v3_large": 960,
}


def build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Returns (feature_extractor, out_channels). The extractor outputs a (B, C, H', W') map
    (final conv stage, before global pooling) so it can feed both an embedding head and Grad-CAM.
    """
    weights_arg = "DEFAULT" if pretrained else None

    if name == "resnet50":
        net = tvm.resnet50(weights=weights_arg)
        extractor = nn.Sequential(*list(net.children())[:-2])  # drop avgpool + fc
    elif name == "efficientnet_b0":
        net = tvm.efficientnet_b0(weights=weights_arg)
        extractor = net.features
    elif name == "mobilenet_v3_large":
        net = tvm.mobilenet_v3_large(weights=weights_arg)
        extractor = net.features
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    return extractor, _BACKBONE_OUT_DIMS[name]


def freeze(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def unfreeze(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True
