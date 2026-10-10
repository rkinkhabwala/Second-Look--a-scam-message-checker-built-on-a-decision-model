import torch

from second_look.config import Settings
from second_look.engines import weights
from second_look.engines.device import pick_device, pick_dtype


def test_explicit_device_wins(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert pick_device("cpu") == "cpu"


def test_auto_device_order(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert pick_device() == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert pick_device() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pick_device() == "cpu"


def test_dtypes(monkeypatch):
    assert pick_dtype("cpu") is torch.float32
    assert pick_dtype("mps") is torch.bfloat16
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert pick_dtype("cuda") is torch.float16


def test_model_refs_are_pinned():
    s = Settings()
    for engine in ("llm", "nli"):
        repo, rev = weights.model_ref(engine, s)
        assert "/" in repo and len(rev) == 40


def test_is_cached_needs_config_and_weights(monkeypatch):
    present = {"config.json", "model.safetensors.index.json"}
    monkeypatch.setattr(weights, "try_to_load_from_cache",
                        lambda repo, name, revision, cache_dir: f"/c/{name}" if name in present else None)
    assert weights.is_cached("org/m", "a" * 40)
    present.discard("model.safetensors.index.json")
    assert not weights.is_cached("org/m", "a" * 40)


def test_download_excludes_pickles_and_onnx():
    assert not any(p.endswith((".bin", ".onnx")) or "onnx" in p for p in weights.ALLOW_PATTERNS)
