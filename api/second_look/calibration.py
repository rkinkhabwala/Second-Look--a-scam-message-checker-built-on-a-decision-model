"""Temperature scaling for local engines (spec §5).

One temperature T per question, fit on a held-out split by minimising
negative log-likelihood, then applied at inference:
- noul: p = sigmoid(z / T), where z is the yes-vs-no logit
- choice / score: probabilities = softmax(logits / T)

Only questions with labels can be fit (is_scam everywhere; scam_type where a
dataset labels it). Others keep T = 1. Jev is used as-is.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from second_look.engines.base import Answer, Choice, EngineResult, Noul, Question
from second_look.engines.llm_scoring import softmax
from second_look.engines.nli import sigmoid

CALIBRATION_DIR = Path(__file__).resolve().parent / "eval" / "calibration"
LOG_T_BOUNDS = (math.log(0.05), math.log(20.0))


# ---------------------------------------------------------------- logits


def answer_logits(answer: Answer) -> list[float] | None:
    """Raw per-option logits saved by a local engine, or None (e.g. Jev answers)."""
    raw = answer.raw or {}
    return raw.get("letter_logits") or raw.get("entailment_scores")


def binary_logit(logits: list[float]) -> float:
    """Yes-vs-no logit: LLM gives [true, false] letter logits, NLI gives one entailment score."""
    return logits[0] - logits[1] if len(logits) == 2 else logits[0]


# ----------------------------------------------------------------- fitting


def _nll_binary(log_t: float, z: np.ndarray, y: np.ndarray) -> float:
    s = z / math.exp(log_t)
    # log(1 + e^-s) for y=1, log(1 + e^s) for y=0, computed stably
    return float(np.mean(np.logaddexp(0, -s) * y + np.logaddexp(0, s) * (1 - y)))


def _nll_multiclass(log_t: float, logits: np.ndarray, y: np.ndarray) -> float:
    s = logits / math.exp(log_t)
    s = s - s.max(axis=1, keepdims=True)
    log_probs = s - np.log(np.exp(s).sum(axis=1, keepdims=True))
    return float(-np.mean(log_probs[np.arange(len(y)), y]))


def fit_temperature_binary(z: list[float] | np.ndarray, y: list[int] | np.ndarray) -> float:
    z, y = np.asarray(z, float), np.asarray(y, float)
    res = minimize_scalar(_nll_binary, bounds=LOG_T_BOUNDS, args=(z, y), method="bounded")
    return float(math.exp(res.x))


def fit_temperature_multiclass(logits: list[list[float]] | np.ndarray, y: list[int] | np.ndarray) -> float:
    logits, y = np.asarray(logits, float), np.asarray(y, int)
    res = minimize_scalar(_nll_multiclass, bounds=LOG_T_BOUNDS, args=(logits, y), method="bounded")
    return float(math.exp(res.x))


# ---------------------------------------------------------------- applying


def apply_temperature(answer: Answer, question: Question, t: float) -> Answer:
    logits = answer_logits(answer)
    if logits is None or t == 1.0:
        return answer
    raw = {**(answer.raw or {}), "temperature": t}
    if isinstance(question, Noul):
        p = sigmoid(binary_logit(logits) / t)
        return Answer(type="noul", value=p, probabilities={"true": p, "false": 1 - p},
                      confidence=abs(2 * p - 1), raw=raw)
    probs = softmax([x / t for x in logits])
    ranked = sorted(probs, reverse=True)
    by_key = dict(zip(answer.probabilities, probs))  # same option order the engine produced
    confidence = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)
    if isinstance(question, Choice):
        top = max(by_key, key=by_key.__getitem__)
        return Answer(type="choice", value=top, probabilities=by_key, confidence=confidence, raw=raw)
    expected = sum(i * p for i, p in enumerate(probs))
    return Answer(type="score", value=expected, probabilities=by_key, confidence=confidence, raw=raw)


@dataclass
class Calibration:
    engine: str
    model: str
    prompt_version: str | None
    temperatures: dict[str, float]
    fit_on: dict = field(default_factory=dict)

    def apply(self, answers: dict[str, Answer], questions: dict[str, Question]) -> dict[str, Answer]:
        return {
            k: apply_temperature(a, questions[k], self.temperatures[k]) if k in self.temperatures else a
            for k, a in answers.items()
        }

    def matches(self, engine) -> bool:
        return engine.name == self.engine and getattr(engine, "model", None) == self.model

    def save(self, directory: Path | None = None) -> Path:
        directory = directory or CALIBRATION_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.engine}.json"
        path.write_text(json.dumps(self.__dict__, indent=2) + "\n")
        return path

    @classmethod
    def load(cls, engine_name: str, directory: Path | None = None) -> Calibration | None:
        path = (directory or CALIBRATION_DIR) / f"{engine_name}.json"
        return cls(**json.loads(path.read_text())) if path.exists() else None


def fit_calibration(
    engine: str,
    model: str,
    prompt_version: str | None,
    answers: list[dict[str, Answer]],
    labels: dict[str, list],
    questions: dict[str, Question],
    *,
    min_examples: int = 30,
) -> Calibration:
    """Fit T for every question with enough labels.

    `labels[key][i]` is the true label for answers[i]: 0/1 for noul, an option key
    for choice, or None when that example has no label for that question.
    """
    temps: dict[str, float] = {}
    counts: dict[str, int] = {}
    for key, ys in labels.items():
        q = questions[key]
        rows = [(answer_logits(a[key]), y) for a, y in zip(answers, ys) if y is not None and key in a]
        rows = [(lg, y) for lg, y in rows if lg is not None]
        if len(rows) < min_examples:
            continue
        if isinstance(q, Noul):
            temps[key] = fit_temperature_binary([binary_logit(lg) for lg, _ in rows], [int(y) for _, y in rows])
        else:
            options = list(q.criteria) if isinstance(q, Choice) else [str(i) for i in range(len(q.criteria))]
            temps[key] = fit_temperature_multiclass([lg for lg, _ in rows], [options.index(y) for _, y in rows])
        counts[key] = len(rows)
    return Calibration(engine, model, prompt_version, temps, {"examples_per_question": counts})


class CalibratedEngine:
    """Wraps a local engine and applies its fitted temperatures to every answer."""

    def __init__(self, inner, calibration: Calibration):
        if not calibration.matches(inner):
            raise ValueError(f"Calibration is for {calibration.engine}/{calibration.model}, not "
                             f"{inner.name}/{getattr(inner, 'model', None)}")
        self.inner = inner
        self.calibration = calibration
        self.name = inner.name
        self.model = inner.model

    def build_payload(self, state, questions):
        return self.inner.build_payload(state, questions)

    async def run(self, state, questions) -> EngineResult:
        r = await self.inner.run(state, questions)
        r.answers = self.calibration.apply(r.answers, questions)
        return r

    async def decide(self, state, questions) -> dict[str, Answer]:
        return (await self.run(state, questions)).answers
