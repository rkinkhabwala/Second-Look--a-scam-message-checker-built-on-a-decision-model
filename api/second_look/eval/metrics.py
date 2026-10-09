"""Metrics for one engine on one split (spec §9). Pure functions over arrays."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def accuracy(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    return float(np.mean((p >= threshold).astype(int) == y))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y, p))


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Equal-width bins over P(scam): mean predicted vs observed scam rate."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        out.append({
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "count": n,
            "mean_predicted": float(p[mask].mean()) if n else None,
            "observed_rate": float(y[mask].mean()) if n else None,
        })
    return out


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: count-weighted |observed rate - mean predicted| per bin."""
    total = len(y)
    return float(sum(
        b["count"] / total * abs(b["observed_rate"] - b["mean_predicted"])
        for b in reliability_bins(y, p, n_bins) if b["count"]
    ))


def _count_for(coverage: float, n: int) -> int:
    """Messages answered at a coverage level; tolerant of float error (0.6 * 5 = 3.0000000000000004)."""
    return max(1, math.ceil(round(coverage * n, 9)))


def coverage_curve(y: np.ndarray, p: np.ndarray, points: int = 101) -> list[dict]:
    """Accuracy vs coverage as a symmetric abstain band around 0.5 widens.

    Messages are answered most-confident first (|p - 0.5| descending); at each
    coverage level the band half-width is the confidence of the last one answered.
    """
    order = np.argsort(-np.abs(p - 0.5), kind="stable")
    correct = ((p[order] >= 0.5).astype(int) == y[order]).astype(float)
    cum_acc = np.cumsum(correct) / np.arange(1, len(y) + 1)
    margin = np.abs(p[order] - 0.5)
    out = []
    for c in np.linspace(0, 1, points)[1:]:
        k = _count_for(c, len(y))
        out.append({
            "coverage": float(k / len(y)),
            "accuracy": float(cum_acc[k - 1]),
            "band": [float(0.5 - margin[k - 1]), float(0.5 + margin[k - 1])],
        })
    return out


def accuracy_at_coverage(y: np.ndarray, p: np.ndarray, coverage: float) -> float:
    order = np.argsort(-np.abs(p - 0.5), kind="stable")
    k = _count_for(coverage, len(y))
    top = order[:k]
    return float(np.mean((p[top] >= 0.5).astype(int) == y[top]))


def recommend_band(y: np.ndarray, p: np.ndarray, target_accuracy: float) -> dict | None:
    """Widest coverage whose answered set still meets the target accuracy (a suggestion only)."""
    best = None
    for point in coverage_curve(y, p, points=len(y) + 1 if len(y) < 1000 else 1001):
        if point["accuracy"] >= target_accuracy:
            best = point
    return best and {"target_accuracy": target_accuracy, **best}


def policy_metrics(y: np.ndarray, verdicts: list[str]) -> dict:
    """How the configured verdict policy (band + risk-confidence gate) actually behaves."""
    answered = [(yi, v) for yi, v in zip(y, verdicts) if v != "UNSURE"]
    correct = sum((v == "LIKELY_SCAM") == bool(yi) for yi, v in answered)
    return {
        "coverage": len(answered) / len(y) if len(y) else 0.0,
        "accuracy_when_answered": correct / len(answered) if answered else None,
        "unsure_rate": 1 - len(answered) / len(y) if len(y) else 0.0,
        "counts": {k: verdicts.count(k) for k in ("LIKELY_SCAM", "UNSURE", "PROBABLY_SAFE")},
    }


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def cost_per_1000(input_tokens: list[int], price_per_mtok: float) -> dict:
    if not input_tokens:
        return {"usd": None, "mean_input_tokens": None, "price_per_mtok_usd": price_per_mtok}
    mean = float(np.mean(input_tokens))
    return {
        "usd": mean * 1000 * price_per_mtok / 1_000_000,
        "mean_input_tokens": mean,
        "price_per_mtok_usd": price_per_mtok,
    }
