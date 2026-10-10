from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from second_look.config import Settings
from second_look.engines.base import EngineError
from second_look.engines.fake import FakeEngine
from second_look.engines.registry import EngineProvider, EngineStatus
from second_look.main import ENGINE_FAILED_MESSAGE, NO_ENGINE_MESSAGE, create_app
from second_look.questions import QUESTIONS
from second_look.storage import Storage
from tests.test_jev import jev_body
from tests.test_verdict import answers


class FakeProvider(EngineProvider):
    """Serves an injected engine under `name` and reports it ready."""

    def __init__(self, engine, settings=Settings(), name="jev"):
        super().__init__(settings)
        self._engines = {name: engine}

    def _status(self, name):
        if name in self._engines:
            return EngineStatus(name, True, None, "fake")
        return super()._status(name)


def make_app(settings=Settings(), provider=None):
    return create_app(settings, provider, Storage("sqlite://"))


def client_with(engine) -> TestClient:
    return TestClient(make_app(provider=FakeProvider(engine)))


SCAM = "E-ZPass: unpaid toll of $4.15. Pay today at https://ezpass.toll-pay.top or face a $50 fine."


def test_check_happy_path():
    engine = FakeEngine(answers(0.93, scam_type="toll_or_fine", flags={"urgency": 0.9}))
    r = client_with(engine).post("/api/check", json={"text": SCAM, "channel": "sms", "claimed_sender": "E-ZPass"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] == "LIKELY_SCAM"
    assert body["probability"] == 0.93
    assert body["scam_type"] == "toll_or_fine"
    assert body["red_flags"] and body["next_steps"]
    assert body["engine"] == "fake"
    assert body["truncated"] is False
    assert isinstance(body["latency_ms"], float)

    state, questions = engine.calls[0]
    assert state["claimed_sender"] == "E-ZPass"
    assert [u["registered_domain"] for u in state["urls"]] == ["toll-pay.top"]
    assert questions == QUESTIONS


def test_check_long_message_reports_truncation():
    engine = FakeEngine(answers(0.1, scam_type="not_a_scam"))
    r = client_with(engine).post("/api/check", json={"text": "hello " * 30_000})
    assert r.status_code == 200
    assert r.json()["truncated"] is True


def test_no_api_key_returns_clear_503():
    c = TestClient(make_app(Settings(typesafe_api_key=None)))
    r = c.post("/api/check", json={"text": SCAM})
    assert r.status_code == 503
    assert r.json()["detail"] == NO_ENGINE_MESSAGE
    assert "TYPESAFE_API_KEY" in r.json()["detail"]


def test_engines_endpoint_reports_readiness():
    no_key = TestClient(make_app(Settings())).get("/api/engines").json()
    assert no_key["default"] is None
    assert [e["name"] for e in no_key["engines"]] == ["jev", "llm", "nli"]
    assert no_key["engines"][0] == {
        "name": "jev", "ready": False, "reason": "TYPESAFE_API_KEY is not set", "model": "jev-1.13.0"
    }
    for local in no_key["engines"][1:]:
        assert local["ready"] is False and "weights download" in local["reason"]
        assert "@" in local["model"]
    with_key = TestClient(make_app(Settings(typesafe_api_key="test-key-not-real"))).get("/api/engines").json()
    assert with_key["engines"][0]["ready"] is True
    assert with_key["default"] == "jev"


def test_unknown_engine_is_400():
    r = client_with(FakeEngine({})).post("/api/check", json={"text": "hi", "engine": "gpt"})
    assert r.status_code == 400


@pytest.mark.parametrize("payload", [{}, {"text": "   "}, {"text": "hi", "channel": "fax"}])
def test_validation_errors(payload):
    assert client_with(FakeEngine({})).post("/api/check", json=payload).status_code == 422


def test_image_not_supported_yet():
    r = client_with(FakeEngine({})).post("/api/check", json={"image": "aGVsbG8="})
    assert r.status_code == 422
    assert "paste" in r.json()["detail"]


def test_engine_failure_is_502_with_template_message():
    def boom(state, qs):
        raise EngineError("Jev returned HTTP 500: secret internals", status=500)

    r = client_with(FakeEngine(boom)).post("/api/check", json={"text": SCAM})
    assert r.status_code == 502
    assert r.json()["detail"] == ENGINE_FAILED_MESSAGE


def test_check_end_to_end_with_real_jev_engine_and_mocked_http(respx_router):
    respx_router.post("https://api.typesafe.ai/v1/systemone").mock(
        return_value=httpx.Response(200, json=jev_body())
    )
    c = TestClient(make_app(Settings(typesafe_api_key="test-key-not-real")))
    r = c.post("/api/check", json={"text": SCAM})
    assert r.status_code == 200, r.text
    assert r.json()["engine"] == "jev"
    assert r.json()["verdict"] == "LIKELY_SCAM"  # p=0.9, risk confidence 0.65


# ------------------------------------------------------------- local engines


def test_falls_back_to_local_engine_and_says_so():
    from second_look.main import FALLBACK_NOTICE

    engine = FakeEngine(answers(0.93))
    engine.name = "llm"
    r = TestClient(make_app(provider=FakeProvider(engine, name="llm"))).post("/api/check", json={"text": SCAM})
    assert r.status_code == 200
    assert r.json()["engine"] == "llm" and r.json()["notice"] == FALLBACK_NOTICE


def test_explicitly_chosen_local_engine_has_no_fallback_notice():
    engine = FakeEngine(answers(0.93))
    engine.name = "nli"
    c = TestClient(make_app(provider=FakeProvider(engine, name="nli")))
    r = c.post("/api/check", json={"text": SCAM, "engine": "nli"})
    assert r.status_code == 200 and r.json()["notice"] is None


def test_jev_check_has_no_notice():
    r = client_with(FakeEngine(answers(0.93))).post("/api/check", json={"text": SCAM})
    assert r.json()["notice"] is None


def test_requested_engine_not_ready_is_503_naming_it():
    r = TestClient(make_app()).post("/api/check", json={"text": SCAM, "engine": "llm"})
    assert r.status_code == 503
    assert "'llm' engine isn't ready" in r.json()["detail"] and "weights download llm" in r.json()["detail"]


def test_default_preference_order(monkeypatch):
    monkeypatch.setattr("second_look.engines.registry.is_cached", lambda repo, rev: True)
    assert EngineProvider(Settings()).default_name() == "llm"
    assert EngineProvider(Settings(typesafe_api_key="test-key-not-real")).default_name() == "jev"
    monkeypatch.setattr("second_look.engines.registry.is_cached", lambda repo, rev: "deberta" in repo.lower())
    assert EngineProvider(Settings()).default_name() == "nli"


def test_provider_builds_local_engine_once(monkeypatch):
    from second_look.engines import registry

    built = []
    monkeypatch.setattr("second_look.engines.registry.is_cached", lambda repo, rev: True)
    monkeypatch.setitem(registry.BUILDERS, "llm", lambda s: built.append(1) or FakeEngine({}, name="llm"))
    p = EngineProvider(Settings())
    assert p.get() is p.get("llm")
    assert built == [1]
