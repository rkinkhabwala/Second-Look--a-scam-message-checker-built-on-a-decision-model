"""Which engines exist, whether each is ready, and one shared instance per engine.

Preference order when no engine is requested: Jev (if its key is set), then the
local LLM engine, then the NLI engine (if their weights are cached locally).
Readiness never triggers a download.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from second_look.config import Settings
from second_look.engines.base import DecisionEngine, EngineUnavailable
from second_look.engines.jev import JevEngine
from second_look.engines.weights import is_cached, model_ref

ENGINE_ORDER = ("jev", "llm", "nli")
DEFAULT_ENGINE = "jev"
LOCAL_ENGINES = ("llm", "nli")


@dataclass(frozen=True)
class EngineStatus:
    name: str
    ready: bool
    reason: str | None
    model: str | None = None


class UnknownEngine(KeyError):
    pass


def _build_llm(settings: Settings) -> DecisionEngine:
    from second_look.engines.hf_scorer import HFOptionScorer
    from second_look.engines.llm_scoring import LLMScoringEngine

    return LLMScoringEngine(HFOptionScorer(settings.llm_model, settings.llm_revision, settings.device))


def _build_nli(settings: Settings) -> DecisionEngine:
    from second_look.engines.hf_scorer import HFNLIScorer
    from second_look.engines.nli import NLIEngine

    return NLIEngine(HFNLIScorer(settings.nli_model, settings.nli_revision, settings.device))


BUILDERS: dict[str, Callable[[Settings], DecisionEngine]] = {
    "jev": JevEngine.from_settings,
    "llm": _build_llm,
    "nli": _build_nli,
}


class EngineProvider:
    """Builds engines lazily and reuses them (HTTP pools, loaded model weights)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._engines: dict[str, DecisionEngine] = {}

    def _status(self, name: str) -> EngineStatus:
        if name == "jev":
            key = bool(self.settings.typesafe_api_key)
            return EngineStatus("jev", key, None if key else "TYPESAFE_API_KEY is not set", self.settings.jev_model)
        repo, rev = model_ref(name, self.settings)
        cached = is_cached(repo, rev)
        reason = None if cached else f"weights not downloaded (python -m second_look.engines.weights download {name})"
        return EngineStatus(name, cached, reason, f"{repo}@{rev[:8]}")

    def status(self) -> list[EngineStatus]:
        return [self._status(name) for name in ENGINE_ORDER]

    def default_name(self) -> str | None:
        """The most preferred ready engine, or None if nothing is ready."""
        for s in self.status():
            if s.ready:
                return s.name
        return None

    def get(self, name: str | None = None) -> DecisionEngine:
        if name is not None and name not in BUILDERS:
            raise UnknownEngine(name)
        if name is None:
            name = self.default_name()
            if name is None:
                raise EngineUnavailable("no engine is ready")
        if name in self._engines:
            return self._engines[name]
        status = self._status(name)
        if not status.ready:
            raise EngineUnavailable(status.reason or f"{name} is not ready")
        engine = BUILDERS[name](self.settings)
        self._engines[name] = engine
        return engine

    async def aclose(self) -> None:
        for engine in self._engines.values():
            close = getattr(engine, "aclose", None)
            if close:
                await close()
        self._engines.clear()
