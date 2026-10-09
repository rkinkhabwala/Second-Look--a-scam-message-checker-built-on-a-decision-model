import json

import httpx
import pytest

from second_look.engines import EngineError, EngineUnavailable
from second_look.engines.jev import JevEngine
from second_look.preprocess import build_state, estimate_tokens
from second_look.questions import QUESTIONS

URL = "https://api.typesafe.ai/v1/systemone"
FAKE_KEY = "test-key-not-real"


def jev_body(model="jev-1.13.0"):
    """A response shaped like docs.typesafe.ai's quickstart example, for our question set."""
    answers = {}
    for key, q in QUESTIONS.items():
        if q.type == "noul":
            answers[key] = {"type": "noul", "noul": 0.9}
        elif q.type == "choice":
            opts = list(q.criteria)
            probs = {o: 0.0 for o in opts}
            probs[opts[0]] = 0.8
            probs[opts[-1]] = 0.2
            answers[key] = {"type": "choice", "choice": opts[0], "confidence": 0.7, "probabilities": probs}
        else:
            answers[key] = {
                "type": "score", "score": 3.2, "confidence": 0.65,
                "legend": {str(i): lvl for i, lvl in enumerate(q.criteria)},
                "probabilities": {"0": 0.0, "1": 0.0, "2": 0.1, "3": 0.6, "4": 0.3},
            }
    return {"model": model, "answers": answers, "usage": {"input_tokens": 812, "output_tokens": 90}}


class SleepRecorder:
    def __init__(self):
        self.delays = []

    async def __call__(self, s):
        self.delays.append(s)


@pytest.fixture
def sleeps():
    return SleepRecorder()


@pytest.fixture
def engine(sleeps):
    return JevEngine(FAKE_KEY, sleep=sleeps)


@pytest.fixture
def state():
    return build_state("USPS: parcel held, pay $1.99 at https://usps.com.track-fee.top")[0]


def test_missing_key_is_unavailable():
    with pytest.raises(EngineUnavailable):
        JevEngine("")


def test_client_configuration(engine):
    assert engine._client.timeout.read == 5.0
    assert engine.model == "jev-1.13.0"


async def test_success_maps_every_answer_type(engine, state, respx_router):
    route = respx_router.post(URL).mock(return_value=httpx.Response(200, json=jev_body()))
    result = await engine.run(state, QUESTIONS)

    assert route.call_count == 1
    req = route.calls.last.request
    assert req.headers["authorization"] == f"Bearer {FAKE_KEY}"
    sent = json.loads(req.content)
    assert sent["model"] == "jev-1.13.0"
    assert sent["state"] == state
    assert set(sent["questions"]) == set(QUESTIONS)
    assert sent["questions"]["scam_type"]["criteria"] == QUESTIONS["scam_type"].criteria

    a = result.answers
    assert set(a) == set(QUESTIONS)
    assert a["is_scam"].value == 0.9
    assert a["is_scam"].probabilities == {"true": 0.9, "false": pytest.approx(0.1)}
    assert a["is_scam"].confidence is None
    assert a["scam_type"].value == "delivery_fee" and a["scam_type"].confidence == 0.7
    assert a["risk_level"].value == 3.2 and a["risk_level"].confidence == 0.65
    assert a["risk_level"].probabilities["3"] == 0.6
    assert result.model == "jev-1.13.0"
    assert result.usage["input_tokens"] == 812
    assert result.attempts == 1 and result.truncated is False


async def test_decide_matches_protocol(engine, state, respx_router):
    respx_router.post(URL).mock(return_value=httpx.Response(200, json=jev_body()))
    answers = await engine.decide(state, QUESTIONS)
    assert answers["is_scam"].type == "noul"


async def test_429_then_success_honors_retry_after(engine, state, sleeps, respx_router):
    route = respx_router.post(URL).mock(side_effect=[
        httpx.Response(429, headers={"retry-after": "2"}, json={"detail": "rate limited"}),
        httpx.Response(200, json=jev_body()),
    ])
    result = await engine.run(state, QUESTIONS)
    assert route.call_count == 2
    assert result.attempts == 2
    assert sleeps.delays == [2.0]


async def test_retry_after_ms_header(engine, state, sleeps, respx_router):
    respx_router.post(URL).mock(side_effect=[
        httpx.Response(429, headers={"retry-after-ms": "250"}),
        httpx.Response(200, json=jev_body()),
    ])
    await engine.run(state, QUESTIONS)
    assert sleeps.delays == [0.25]


async def test_529_overloaded_then_success_uses_backoff(engine, state, sleeps, respx_router):
    respx_router.post(URL).mock(side_effect=[
        httpx.Response(529), httpx.Response(503), httpx.Response(200, json=jev_body()),
    ])
    result = await engine.run(state, QUESTIONS)
    assert result.attempts == 3
    assert 0.5 <= sleeps.delays[0] <= 0.625 and 1.0 <= sleeps.delays[1] <= 1.25


async def test_retries_exhausted_on_5xx(engine, state, sleeps, respx_router):
    route = respx_router.post(URL).mock(return_value=httpx.Response(500))
    with pytest.raises(EngineError) as exc:
        await engine.run(state, QUESTIONS)
    assert exc.value.status == 500
    assert route.call_count == 4  # 1 + 3 retries
    assert len(sleeps.delays) == 3


async def test_timeout_every_attempt(engine, state, sleeps, respx_router):
    route = respx_router.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(EngineError, match="timed out after 4 attempts"):
        await engine.run(state, QUESTIONS)
    assert route.call_count == 4


async def test_timeout_then_success(engine, state, respx_router):
    respx_router.post(URL).mock(side_effect=[
        httpx.ConnectTimeout("slow"), httpx.Response(200, json=jev_body()),
    ])
    assert (await engine.run(state, QUESTIONS)).attempts == 2


@pytest.mark.parametrize("status", [400, 401, 403, 422])
async def test_4xx_fails_fast_without_retry(engine, state, sleeps, respx_router, status):
    route = respx_router.post(URL).mock(return_value=httpx.Response(
        status, headers={"x-typesafe-request-id": "req_123"}, json={"detail": "bad field: state"}
    ))
    with pytest.raises(EngineError) as exc:
        await engine.run(state, QUESTIONS)
    assert route.call_count == 1 and sleeps.delays == []
    assert exc.value.status == status and exc.value.request_id == "req_123"
    assert "bad field" in str(exc.value)
    assert FAKE_KEY not in str(exc.value)


async def test_over_limit_message_is_middle_truncated(engine, respx_router):
    text = "URGENT toll notice. " + ("filler words here. " * 12_000) + "Pay at https://toll-pay.top today."
    state, meta = build_state(text)  # no budget here: the engine must enforce Jev's limit itself
    assert estimate_tokens(state) > 32_000 and not meta.truncated

    route = respx_router.post(URL).mock(return_value=httpx.Response(200, json=jev_body()))
    result = await engine.run(state, QUESTIONS)

    assert result.truncated is True
    assert route.call_count == 1
    sent = json.loads(route.calls.last.request.content)
    longest_q = max(estimate_tokens(q) for q in sent["questions"].values())
    assert estimate_tokens(sent["state"]) + longest_q <= 32_000
    assert estimate_tokens(sent) <= 64_000
    msg = sent["state"]["message"]
    assert msg.startswith("URGENT toll notice.") and msg.endswith("today.")
    assert "characters removed" in msg
    assert sent["state"]["urls"] == state["urls"]


async def test_under_limit_state_is_not_touched(engine, state, respx_router):
    route = respx_router.post(URL).mock(return_value=httpx.Response(200, json=jev_body()))
    result = await engine.run(state, QUESTIONS)
    assert result.truncated is False
    assert json.loads(route.calls.last.request.content)["state"] == state


async def test_malformed_response_is_engine_error(engine, state, respx_router):
    body = jev_body()
    del body["answers"]["urgency"]
    respx_router.post(URL).mock(return_value=httpx.Response(200, json=body))
    with pytest.raises(EngineError, match="urgency"):
        await engine.run(state, QUESTIONS)
