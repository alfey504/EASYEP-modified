from pathlib import Path

import pytest
import torch

from moe_pruning.config import ConfigError, load_config
from moe_pruning.utils import device as device_utils


def _cfg(tmp_path: Path, **overrides):
    env = tmp_path / ".env"
    env.write_text("C4_SUPPORT_KEYWORDS=customer support,FAQ\n")
    return load_config(env, {k: str(v) for k, v in overrides.items()})


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Real environment variables override .env; keep tests hermetic.
    import os

    for key in list(os.environ):
        if key.isupper() and key.startswith(("MODEL_", "HF_", "DEVICE", "PRUNE_", "BIAS_", "EVAL_", "C4_")):
            monkeypatch.delenv(key, raising=False)


def test_defaults(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.hf_model_id == "google/switch-base-8"
    assert cfg.model_path == ""
    assert cfg.model_source == "google/switch-base-8"
    assert cfg.prune_experts_per_layer == 2
    assert cfg.prune_layers == "first"
    assert cfg.model_dtype == "float32"
    assert cfg.c4_support_keywords == ("customer support", "FAQ")


def test_overrides_and_types(tmp_path):
    cfg = _cfg(tmp_path, PRUNE_EXPERTS_PER_LAYER=3, BIAS_LEARNING_RATE="0.01", DEVICE="CPU")
    assert cfg.prune_experts_per_layer == 3
    assert cfg.bias_learning_rate == pytest.approx(0.01)
    assert cfg.device == "cpu"


def test_invalid_local_model_path_raises(tmp_path):
    with pytest.raises(ConfigError, match="MODEL_PATH"):
        _cfg(tmp_path, MODEL_PATH=str(tmp_path / "missing"))


def test_valid_local_model_path(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    cfg = _cfg(tmp_path, MODEL_PATH=str(model_dir))
    assert cfg.model_source == str(model_dir)


def test_all_errors_reported_together(tmp_path):
    with pytest.raises(ConfigError) as exc:
        _cfg(tmp_path, PRUNE_EXPERTS_PER_LAYER=0, DEVICE="tpu", MODEL_DTYPE="int8")
    message = str(exc.value)
    assert "PRUNE_EXPERTS_PER_LAYER" in message and "DEVICE" in message and "MODEL_DTYPE" in message


def test_device_auto_priority(monkeypatch):
    monkeypatch.setattr(device_utils, "is_available", lambda kind: kind in ("cpu", "mps"))
    assert device_utils.resolve_device("auto").type == "mps"
    monkeypatch.setattr(device_utils, "is_available", lambda kind: True)
    assert device_utils.resolve_device("auto").type == "cuda"
    monkeypatch.setattr(device_utils, "is_available", lambda kind: kind == "cpu")
    assert device_utils.resolve_device("auto").type == "cpu"


def test_explicit_unavailable_device_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(device_utils, "is_available", lambda kind: kind == "cpu")
    with pytest.raises(device_utils.DeviceUnavailableError):
        device_utils.resolve_device("cuda")
    with pytest.raises(device_utils.DeviceUnavailableError):
        device_utils.resolve_device("mps")
    assert device_utils.resolve_device("cpu") == torch.device("cpu")


def test_dtype_resolution():
    assert device_utils.resolve_dtype("float32") is torch.float32
    with pytest.raises(ValueError):
        device_utils.resolve_dtype("int8")
