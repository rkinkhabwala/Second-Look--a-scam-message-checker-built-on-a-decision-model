"""FastAPI app (spec §8)."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from second_look.config import Settings, get_settings
from second_look.engines.base import EngineError, EngineUnavailable
from second_look.engines.jev import JevEngine
from second_look.engines.registry import DEFAULT_ENGINE, EngineProvider, UnknownEngine
from second_look.preprocess import Channel, build_state, state_token_budget
from second_look.questions import QUESTIONS
from second_look.verdict import Verdict, make_verdict

log = logging.getLogger("second_look")

NO_ENGINE_MESSAGE = (
    "No decision engine is configured. Set TYPESAFE_API_KEY (see .env.example) to use Jev; "
    "local engines arrive in a later release."
)
ENGINE_FAILED_MESSAGE = "The decision engine didn't respond properly. Please try again in a moment."
SCREENSHOTS_LATER_MESSAGE = "Screenshot checks aren't available yet. Please paste the message text instead."


class CheckRequest(BaseModel):
    text: str | None = Field(default=None, max_length=200_000)
    image: str | None = None  # base64; Phase 3
    channel: Channel = "sms"
    claimed_sender: str | None = Field(default=None, max_length=200)
    engine: str | None = None
    help_improve: bool = False


class CheckResponse(Verdict):
    engine: str
    latency_ms: float
    truncated: bool


def create_app(settings: Settings | None = None, provider: EngineProvider | None = None) -> FastAPI:
    settings = settings or get_settings()
    provider = provider or EngineProvider(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await provider.aclose()

    app = FastAPI(title="Second Look", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.provider = provider

    def get_provider(request: Request) -> EngineProvider:
        return request.app.state.provider

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/engines")
    def engines(p: EngineProvider = Depends(get_provider)) -> dict:
        return {"default": DEFAULT_ENGINE, "engines": [s.__dict__ for s in p.status()]}

    @app.post("/api/check", response_model=CheckResponse)
    async def check(req: CheckRequest, p: EngineProvider = Depends(get_provider)) -> CheckResponse:
        if req.image is not None:
            raise HTTPException(422, SCREENSHOTS_LATER_MESSAGE)
        if not req.text or not req.text.strip():
            raise HTTPException(422, "Please paste the message you want to check.")
        try:
            engine = p.get(req.engine)
        except UnknownEngine:
            raise HTTPException(400, f"Unknown engine '{req.engine}'.") from None
        except EngineUnavailable:
            raise HTTPException(503, NO_ENGINE_MESSAGE) from None

        budget = state_token_budget(
            {k: q.model_dump() for k, q in QUESTIONS.items()},
            settings.jev_max_state_tokens,
            settings.jev_max_request_tokens,
        )
        state, meta = build_state(
            req.text, req.channel, req.claimed_sender, max_state_tokens=budget
        )
        start = time.perf_counter()
        try:
            if isinstance(engine, JevEngine):
                result = await engine.run(state, QUESTIONS)
                answers, engine_truncated = result.answers, result.truncated
            else:
                answers, engine_truncated = await engine.decide(state, QUESTIONS), False
        except EngineError as exc:
            log.warning("engine %s failed: %s (status=%s)", engine.name, exc, exc.status)
            raise HTTPException(502, ENGINE_FAILED_MESSAGE) from None
        latency_ms = (time.perf_counter() - start) * 1000

        v = make_verdict(answers, settings)
        return CheckResponse(
            **v.model_dump(),
            engine=engine.name,
            latency_ms=round(latency_ms, 1),
            truncated=meta.truncated or engine_truncated,
        )

    return app


app = create_app()
