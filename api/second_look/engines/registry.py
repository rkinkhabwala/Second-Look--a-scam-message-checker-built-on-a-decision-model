"""Which engines exist, whether each is ready, and one shared instance per engine."""

from __future__ import annotations

from dataclasses import dataclass

from second_look.config import Settings
from second_look.engines.base import DecisionEngine, EngineUnavailable
from second_look.engines.jev import JevEngine

DEFAULT_ENGINE = "jev"


@dataclass(frozen=True)
class EngineStatus:
    name: str
    ready: bool
    reason: str | None
    model: str | None = None


class UnknownEngine(KeyError):
    pass


class EngineProvider:
    """Builds engines lazily and reuses them (and their HTTP connection pools)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._engines: dict[str, DecisionEngine] = {}

    def status(self) -> list[EngineStatus]:
        key = bool(self.settings.typesafe_api_key)
        return [
            EngineStatus(
                name="jev",
                ready=key,
                reason=None if key else "TYPESAFE_API_KEY is not set",
                model=self.settings.jev_model,
            )
        ]

    def get(self, name: str | None = None) -> DecisionEngine:
        name = name or DEFAULT_ENGINE
        if name in self._engines:
            return self._engines[name]
        if name == "jev":
            if not self.settings.typesafe_api_key:
                raise EngineUnavailable("TYPESAFE_API_KEY is not set")
            engine = JevEngine.from_settings(self.settings)
        else:
            raise UnknownEngine(name)
        self._engines[name] = engine
        return engine

    async def aclose(self) -> None:
        for engine in self._engines.values():
            close = getattr(engine, "aclose", None)
            if close:
                await close()
        self._engines.clear()
