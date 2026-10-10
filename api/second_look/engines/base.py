"""The decision engine interface (spec §5).

Mirrors Jev's three question types so every engine is interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel


class Noul(BaseModel):  # yes/no -> P(true)
    type: Literal["noul"] = "noul"
    instructions: str


class Choice(BaseModel):  # one of N options
    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str]  # option -> when it applies


class Score(BaseModel):  # ordered scale
    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str]  # lowest -> highest


Question = Noul | Choice | Score


class Answer(BaseModel):
    type: str
    value: str | float  # chosen option, P(true), or expected level
    probabilities: dict[str, float]
    confidence: float | None
    raw: dict | None = None


@runtime_checkable
class DecisionEngine(Protocol):
    name: str

    async def decide(
        self, state: str | dict, questions: dict[str, Noul | Choice | Score]
    ) -> dict[str, Answer]: ...


@dataclass
class EngineResult:
    """Answers plus run metadata, for engines that offer `run()` alongside `decide()`."""

    answers: dict[str, Answer]
    model: str | None
    usage: dict = field(default_factory=dict)
    truncated: bool = False
    attempts: int = 1
    latency_ms: float = 0.0


class EngineError(Exception):
    """The engine was reachable but could not produce answers (bad request, retries exhausted...)."""

    def __init__(self, message: str, *, status: int | None = None, request_id: str | None = None):
        super().__init__(message)
        self.status = status
        self.request_id = request_id


class EngineUnavailable(EngineError):
    """The engine is not configured (e.g. missing API key) or not ready."""
