"""A scripted engine for tests: returns fixed answers, records every call."""

from __future__ import annotations

from collections.abc import Callable

from second_look.engines.base import Answer, Choice, Noul, Score


def noul(p: float) -> Answer:
    return Answer(type="noul", value=p, probabilities={"true": p, "false": 1 - p}, confidence=None)


def choice(probabilities: dict[str, float], confidence: float | None = None) -> Answer:
    ranked = sorted(probabilities.values(), reverse=True)
    if confidence is None:
        confidence = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)
    top = max(probabilities, key=probabilities.__getitem__)
    return Answer(type="choice", value=top, probabilities=probabilities, confidence=confidence)


def score(probabilities: list[float], confidence: float | None = None) -> Answer:
    """Level probabilities lowest -> highest; value is the probability-weighted level index."""
    ranked = sorted(probabilities, reverse=True)
    if confidence is None:
        confidence = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)
    expected = sum(i * p for i, p in enumerate(probabilities))
    return Answer(
        type="score",
        value=expected,
        probabilities={str(i): p for i, p in enumerate(probabilities)},
        confidence=confidence,
    )


Script = dict[str, Answer] | Callable[[str | dict, dict], dict[str, Answer]]


class FakeEngine:
    def __init__(self, script: Script, name: str = "fake"):
        self.name = name
        self._script = script
        self.calls: list[tuple[str | dict, dict]] = []

    async def decide(
        self, state: str | dict, questions: dict[str, Noul | Choice | Score]
    ) -> dict[str, Answer]:
        self.calls.append((state, questions))
        answers = self._script(state, questions) if callable(self._script) else self._script
        return {k: answers[k] for k in questions if k in answers}
