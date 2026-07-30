import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sigverify.data.synthetic import generate_synthetic_signature_image, generate_synthetic_stroke
from sigverify.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def default_config():
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    # Keep unit tests fast and CPU-only regardless of the host machine.
    cfg["device"] = "cpu"
    cfg["static_branch"]["pretrained"] = False
    cfg["static_branch"]["backbone"] = "mobilenet_v3_large"
    return cfg


@pytest.fixture
def sample_image():
    return generate_synthetic_signature_image("writer_000", sample_seed=1, forged=False, size=(120, 360))


@pytest.fixture
def sample_forged_image():
    return generate_synthetic_signature_image("writer_000", sample_seed=2, forged=True, size=(120, 360))


@pytest.fixture
def sample_stroke():
    return generate_synthetic_stroke("writer_000", sample_seed=1, forged=False, num_points=64)


@pytest.fixture
def sample_forged_stroke():
    return generate_synthetic_stroke("writer_000", sample_seed=2, forged=True, num_points=64)
