"""Pick the torch device and dtype for local engines."""

from __future__ import annotations

import torch


def pick_device(preferred: str | None = None) -> str:
    """An explicit choice wins; otherwise Apple GPU (mps), then CUDA, then CPU."""
    if preferred:
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    """Half precision on GPUs (bf16 where supported, else fp16); full precision on CPU."""
    if device == "cpu":
        return torch.float32
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float16
    return torch.bfloat16
