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
    """Half precision on GPUs, full precision on CPU.

    Apple GPUs get fp16: measured on an M1 Pro, bf16 was 60% slower (no native
    bf16) and drifted further from the reference forward pass.
    """
    if device == "cpu":
        return torch.float32
    if device == "mps":
        return torch.float16
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float16
    return torch.bfloat16
