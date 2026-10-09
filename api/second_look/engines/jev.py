"""TypeSafe Jev engine: POST /v1/systemone, one request per message with every question.

API reference: https://docs.typesafe.ai/api.md (checked 2026-10-09, see PLAN.md §1).
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx

from second_look.config import Settings
from second_look.engines.base import Answer, Choice, EngineError, EngineUnavailable, Noul, Score
from second_look.preprocess import estimate_tokens, state_token_budget, truncate_middle

RETRY_STATUSES = frozenset({408, 429})  # plus every 5xx (529 = Overloaded)
MAX_RETRY_AFTER_S = 10.0


@dataclass
class JevResult:
    answers: dict[str, Answer]
    model: str | None
    usage: dict = field(default_factory=dict)
    truncated: bool = False
    attempts: int = 1
    latency_ms: float = 0.0


class JevEngine:
    name = "jev"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "jev-1.13.0",
        base_url: str = "https://api.typesafe.ai",
        timeout_s: float = 5.0,
        max_retries: int = 3,
        max_request_tokens: int = 64_000,
        max_state_tokens: int = 32_000,
        backoff_initial_s: float = 0.5,
        backoff_max_s: float = 4.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        client: httpx.AsyncClient | None = None,
    ):
        if not api_key:
            raise EngineUnavailable("TYPESAFE_API_KEY is not set")
        self.model = model
        self.max_retries = max_retries
        self.max_request_tokens = max_request_tokens
        self.max_state_tokens = max_state_tokens
        self.backoff_initial_s = backoff_initial_s
        self.backoff_max_s = backoff_max_s
        self._sleep = sleep
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_s),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs) -> JevEngine:
        return cls(
            settings.typesafe_api_key or "",
            model=settings.jev_model,
            base_url=settings.jev_base_url,
            timeout_s=settings.jev_timeout_s,
            max_retries=settings.jev_max_retries,
            max_request_tokens=settings.jev_max_request_tokens,
            max_state_tokens=settings.jev_max_state_tokens,
            **kwargs,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def decide(
        self, state: str | dict, questions: dict[str, Noul | Choice | Score]
    ) -> dict[str, Answer]:
        return (await self.run(state, questions)).answers

    def build_payload(
        self, state: str | dict, questions: dict[str, Noul | Choice | Score]
    ) -> tuple[dict, bool]:
        """The exact request body, with the state cut to fit Jev's token limits."""
        q_json = {k: q.model_dump() for k, q in questions.items()}
        state, truncated = self._fit_state(state, q_json)
        return {"model": self.model, "state": state, "questions": q_json}, truncated

    async def run(self, state: str | dict, questions: dict[str, Noul | Choice | Score]) -> JevResult:
        payload, truncated = self.build_payload(state, questions)
        start = time.perf_counter()
        response, attempts = await self._post_with_retries(payload)
        latency_ms = (time.perf_counter() - start) * 1000
        body = response.json()
        return JevResult(
            answers=self._map_answers(body, questions),
            model=body.get("model"),
            usage=body.get("usage") or {},
            truncated=truncated,
            attempts=attempts,
            latency_ms=latency_ms,
        )

    # ---------------------------------------------------------------- limits

    def state_budget(self, q_json: dict) -> int:
        return state_token_budget(q_json, self.max_state_tokens, self.max_request_tokens)

    def _fit_state(self, state: str | dict, q_json: dict) -> tuple[str | dict, bool]:
        """Jev limits: 64K tokens per request; 32K for state + the longest question."""
        budget = self.state_budget(q_json)
        if estimate_tokens(state) <= budget:
            return state, False
        if isinstance(state, str):
            return truncate_middle(state, budget)[0], True
        if isinstance(state, dict) and isinstance(state.get("message"), str):
            overhead = estimate_tokens({**state, "message": ""})
            message, _ = truncate_middle(state["message"], budget - overhead)
            fitted = {**state, "message": message}
            if estimate_tokens(fitted) <= budget:
                return fitted, True
        raise EngineError("State is too large for Jev's token limits even after truncation")

    # --------------------------------------------------------------- HTTP

    async def _post_with_retries(self, payload: dict) -> tuple[httpx.Response, int]:
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            last = attempt == attempts
            try:
                response = await self._client.post("/v1/systemone", json=payload)
            except httpx.TimeoutException as exc:
                if last:
                    raise EngineError(f"Jev timed out after {attempt} attempts") from exc
                await self._sleep(self._backoff(attempt))
                continue
            except httpx.TransportError as exc:
                if last:
                    raise EngineError(f"Could not reach Jev: {type(exc).__name__}") from exc
                await self._sleep(self._backoff(attempt))
                continue

            if response.is_success:
                return response, attempt
            status = response.status_code
            request_id = response.headers.get("x-typesafe-request-id")
            retryable = status in RETRY_STATUSES or status >= 500
            if not retryable or last:
                raise EngineError(
                    f"Jev returned HTTP {status}: {_error_detail(response)}",
                    status=status,
                    request_id=request_id,
                )
            await self._sleep(self._retry_after(response) or self._backoff(attempt))
        raise AssertionError("unreachable")

    def _backoff(self, attempt: int) -> float:
        base = min(self.backoff_max_s, self.backoff_initial_s * 2 ** (attempt - 1))
        return base * (1 + random.uniform(0, 0.25))

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        try:
            if ms := response.headers.get("retry-after-ms"):
                return min(float(ms) / 1000, MAX_RETRY_AFTER_S)
            if s := response.headers.get("retry-after"):
                return min(float(s), MAX_RETRY_AFTER_S)
        except ValueError:  # HTTP-date form: fall back to our own backoff
            return None
        return None

    # ------------------------------------------------------------- mapping

    @staticmethod
    def _map_answers(body: dict, questions: dict[str, Noul | Choice | Score]) -> dict[str, Answer]:
        raw_answers = body.get("answers")
        if not isinstance(raw_answers, dict):
            raise EngineError("Jev response has no 'answers' object")
        out: dict[str, Answer] = {}
        for key, question in questions.items():
            a = raw_answers.get(key)
            if not isinstance(a, dict):
                raise EngineError(f"Jev response is missing the answer for '{key}'")
            if a.get("type") != question.type:
                raise EngineError(f"Jev answered '{key}' as {a.get('type')!r}, expected {question.type!r}")
            try:
                if question.type == "noul":
                    p = float(a["noul"])
                    out[key] = Answer(
                        type="noul", value=p, probabilities={"true": p, "false": 1 - p},
                        confidence=None, raw=a,
                    )
                elif question.type == "choice":
                    out[key] = Answer(
                        type="choice", value=a["choice"],
                        probabilities={k: float(v) for k, v in a["probabilities"].items()},
                        confidence=a.get("confidence"), raw=a,
                    )
                else:
                    out[key] = Answer(
                        type="score", value=float(a["score"]),
                        probabilities={str(k): float(v) for k, v in a["probabilities"].items()},
                        confidence=a.get("confidence"), raw=a,
                    )
            except (KeyError, TypeError, ValueError) as exc:
                raise EngineError(f"Jev answer for '{key}' is malformed: {exc!r}") from exc
        return out


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200] or response.reason_phrase
    if isinstance(body, dict):
        detail = body.get("detail") or body.get("error") or body.get("message") or body
        return str(detail)[:300]
    return str(body)[:300]
