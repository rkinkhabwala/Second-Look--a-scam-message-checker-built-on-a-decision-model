"""Open-source engine B: zero-shot NLI (spec §5). The fast CPU/GPU baseline.

Each option becomes a hypothesis about the message (questions.nli_hypotheses).
The model scores premise/hypothesis pairs; we use entailment minus
not-entailment logits as each option's score.
- noul: sigmoid of its single hypothesis's score = P(true)
- choice / score: softmax of the options' scores
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from typing import Protocol

from second_look.engines.base import Answer, Choice, EngineResult, Noul, Question
from second_look.engines.llm_scoring import softmax
from second_look.engines.render import render_state

PROMPT_VERSION = "nli-v2"  # v2: premise cut to 600 chars
# Each of the ~35 hypotheses re-reads the premise, so premise length drives speed: at 1,400 chars
# long emails took ~14 s per check on an M1 Pro. 600 chars (~150 tokens) keeps the opening and the
# call to action (middle-truncated) for a fraction of the cost.
MAX_MESSAGE_CHARS = 600


def middle_truncate_chars(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    half = max_chars // 2
    return f"{text[:half]} […] {text[-half:]}", True


class NLIScorer(Protocol):
    model_id: str

    def entailment_scores(self, premise: str, hypotheses: list[str]) -> list[float]:
        """Entailment-minus-not-entailment logit for each hypothesis against the premise."""
        ...


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def to_answer(question: Question, options: list[str], scores: list[float]) -> Answer:
    raw = {"entailment_scores": scores}
    if isinstance(question, Noul):
        p = sigmoid(scores[0])
        return Answer(type="noul", value=p, probabilities={"true": p, "false": 1 - p},
                      confidence=abs(2 * p - 1), raw=raw)
    probs = softmax(scores)
    ranked = sorted(probs, reverse=True)
    confidence = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)
    by_key = dict(zip(options, probs))
    if isinstance(question, Choice):
        top = max(by_key, key=by_key.__getitem__)
        return Answer(type="choice", value=top, probabilities=by_key, confidence=confidence, raw=raw)
    expected = sum(i * p for i, p in enumerate(probs))
    return Answer(type="score", value=expected, probabilities=by_key, confidence=confidence, raw=raw)


class NLIEngine:
    name = "nli"

    def __init__(self, scorer: NLIScorer, *, max_message_chars: int = MAX_MESSAGE_CHARS):
        self.scorer = scorer
        self.model = scorer.model_id
        self.max_message_chars = max_message_chars
        self._lock = threading.Lock()

    def _premise(self, state: str | dict) -> tuple[str, bool]:
        if isinstance(state, str):
            text, truncated = middle_truncate_chars(state, self.max_message_chars)
            return render_state(text), truncated
        message, truncated = middle_truncate_chars(state.get("message", ""), self.max_message_chars)
        return render_state({**state, "message": message}), truncated

    @staticmethod
    def _hypotheses(questions: dict[str, Question]) -> dict[str, dict[str, str]]:
        from second_look.questions import nli_hypotheses

        return {key: nli_hypotheses(key) for key in questions}

    def build_payload(self, state: str | dict, questions: dict[str, Question]) -> tuple[dict, bool]:
        premise, truncated = self._premise(state)
        return {"engine": self.name, "model": self.model, "prompt_version": PROMPT_VERSION,
                "premise": premise, "hypotheses": self._hypotheses(questions)}, truncated

    def _score(self, state: str | dict, questions: dict[str, Question]) -> EngineResult:
        premise, truncated = self._premise(state)
        hyps = self._hypotheses(questions)
        flat = [h for per_q in hyps.values() for h in per_q.values()]
        start = time.perf_counter()
        with self._lock:
            scores = self.scorer.entailment_scores(premise, flat)
        latency_ms = (time.perf_counter() - start) * 1000
        answers, i = {}, 0
        for key, per_q in hyps.items():
            n = len(per_q)
            answers[key] = to_answer(questions[key], list(per_q), scores[i : i + n])
            i += n
        return EngineResult(answers=answers, model=self.model, truncated=truncated, latency_ms=latency_ms)

    async def run(self, state: str | dict, questions: dict[str, Question]) -> EngineResult:
        return await asyncio.to_thread(self._score, state, questions)

    async def decide(self, state: str | dict, questions: dict[str, Question]) -> dict[str, Answer]:
        return (await self.run(state, questions)).answers
