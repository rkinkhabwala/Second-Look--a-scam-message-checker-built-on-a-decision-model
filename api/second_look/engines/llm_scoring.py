"""Open-source engine A: LLM option scoring (spec §5).

For each question the model sees the message, the question and lettered
options; we read the next-token log-probabilities of the option letters and
softmax them. No text is generated.

Speed: all questions share the same prefix (system prompt + message), so the
HF scorer encodes that prefix once and scores every question's suffix in one
batched forward pass on top of the cached prefix.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from second_look.engines.base import Answer, Choice, EngineResult, Noul, Question, Score
from second_look.engines.render import render_state
from second_look.preprocess import truncate_middle

PROMPT_VERSION = "llm-v1"  # bump when prompts change, so cached eval answers are invalidated
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM_PROMPT = (
    "You are a careful fraud analyst. You read one message and answer a multiple-choice question "
    "about it. Reply with the letter of the single best option and nothing else."
)
NOUL_OPTIONS = ("true", "false")  # option keys for yes/no questions; A = true


# ------------------------------------------------------------------- prompts


@dataclass(frozen=True)
class QuestionPrompt:
    key: str
    question: Question
    options: list[str]  # option keys in letter order
    text: str  # the user-turn text that follows the message


def option_keys(q: Question) -> list[str]:
    if isinstance(q, Noul):
        return list(NOUL_OPTIONS)
    if isinstance(q, Choice):
        return list(q.criteria)
    return [str(i) for i in range(len(q.criteria))]


def question_text(q: Question) -> str:
    if isinstance(q, Noul):
        lines = [f"Statement: {q.instructions}", "Is this statement true for the message above?",
                 "A. True", "B. False"]
    elif isinstance(q, Choice):
        lines = [f"Question: {q.instructions}", "Options:"]
        lines += [f"{LETTERS[i]}. {desc}" for i, desc in enumerate(q.criteria.values())]
    else:
        lines = [f"Question: {q.instructions}", "Options, from lowest to highest:"]
        lines += [f"{LETTERS[i]}. {level}" for i, level in enumerate(q.criteria)]
    lines.append("Answer with a single letter.")
    return "\n".join(lines)


def build_prompts(questions: dict[str, Question]) -> list[QuestionPrompt]:
    out = []
    for key, q in questions.items():
        keys = option_keys(q)
        if len(keys) > len(LETTERS):
            raise ValueError(f"Question '{key}' has more than {len(LETTERS)} options")
        out.append(QuestionPrompt(key=key, question=q, options=keys, text=question_text(q)))
    return out


# -------------------------------------------------------------- probabilities


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def to_answer(prompt: QuestionPrompt, logits: list[float]) -> Answer:
    """Letter logits -> Answer, with confidence = top probability minus second (spec §5)."""
    probs = softmax(logits)
    ranked = sorted(probs, reverse=True)
    confidence = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)
    by_key = dict(zip(prompt.options, probs))
    raw = {"letter_logits": logits}
    q = prompt.question
    if isinstance(q, Noul):
        return Answer(type="noul", value=by_key["true"], probabilities=by_key, confidence=confidence, raw=raw)
    if isinstance(q, Choice):
        top = max(by_key, key=by_key.__getitem__)
        return Answer(type="choice", value=top, probabilities=by_key, confidence=confidence, raw=raw)
    expected = sum(i * p for i, p in enumerate(probs))  # same definition as Jev's score
    return Answer(type="score", value=expected, probabilities=by_key, confidence=confidence, raw=raw)


# --------------------------------------------------------------------- scorer


class OptionScorer(Protocol):
    model_id: str

    def letter_logits(self, system: str, context: str, suffixes: list[str], n_options: list[int]) -> list[list[float]]:
        """For each suffix, the next-token logits of letters A.. (n_options[i] of them) after
        system + context + suffix, formatted as one chat turn."""
        ...


class LLMScoringEngine:
    name = "llm"

    def __init__(self, scorer: OptionScorer, *, max_message_tokens: int = 2_000):
        self.scorer = scorer
        self.model = scorer.model_id
        self.max_message_tokens = max_message_tokens
        self._lock = threading.Lock()  # one model, one forward pass at a time

    def _fit(self, state: str | dict) -> tuple[str | dict, bool]:
        if isinstance(state, str):
            return truncate_middle(state, self.max_message_tokens)
        message, truncated = truncate_middle(state.get("message", ""), self.max_message_tokens)
        return {**state, "message": message}, truncated

    def build_payload(self, state: str | dict, questions: dict[str, Question]) -> tuple[dict, bool]:
        """What determines the output: cache keys built from this change with model or prompt."""
        fitted, truncated = self._fit(state)
        prompts = build_prompts(questions)
        return {
            "engine": self.name,
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "context": render_state(fitted),
            "questions": {p.key: p.text for p in prompts},
        }, truncated

    def _score(self, state: str | dict, questions: dict[str, Question]) -> EngineResult:
        fitted, truncated = self._fit(state)
        prompts = build_prompts(questions)
        start = time.perf_counter()
        with self._lock:
            logits = self.scorer.letter_logits(
                SYSTEM_PROMPT, render_state(fitted), [p.text for p in prompts], [len(p.options) for p in prompts]
            )
        latency_ms = (time.perf_counter() - start) * 1000
        answers = {p.key: to_answer(p, row) for p, row in zip(prompts, logits)}
        return EngineResult(answers=answers, model=self.model, truncated=truncated, latency_ms=latency_ms)

    async def run(self, state: str | dict, questions: dict[str, Question]) -> EngineResult:
        return await asyncio.to_thread(self._score, state, questions)

    async def decide(self, state: str | dict, questions: dict[str, Question]) -> dict[str, Answer]:
        return (await self.run(state, questions)).answers
