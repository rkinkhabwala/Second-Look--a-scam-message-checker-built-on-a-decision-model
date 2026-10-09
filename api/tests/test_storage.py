import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from second_look.config import Settings
from second_look.engines.fake import FakeEngine
from second_look.main import create_app
from second_look.storage import CheckRecord, FeedbackRecord, Storage, mask_pii
from tests.test_api import FakeProvider
from tests.test_verdict import answers

MSG = "Hi it's Mom, new phone! Text me at (555) 123-4567 or mom.help@gmail.com. Reply to 7726 if lost."


@pytest.fixture
def storage():
    return Storage("sqlite://")


@pytest.fixture
def client(storage):
    engine = FakeEngine(answers(0.9, scam_type="family_emergency"))
    return TestClient(create_app(Settings(), FakeProvider(engine), storage))


def rows(storage, model):
    with Session(storage.engine) as s:
        return s.exec(select(model)).all()


def test_mask_pii():
    assert mask_pii(MSG) == (
        "Hi it's Mom, new phone! Text me at [phone] or [email]. Reply to 7726 if lost."
    )
    assert mask_pii("call +44 20 7946 0958 now") == "call [phone] now"


def test_storage_is_lazy(tmp_path):
    db = tmp_path / "x.db"
    Storage(f"sqlite:///{db}")
    assert not db.exists()


def test_default_logs_only_verdict_engine_latency_truncation(client, storage):
    r = client.post("/api/check", json={"text": MSG, "claimed_sender": "Mom"})
    assert r.status_code == 200
    (row,) = rows(storage, CheckRecord)
    assert row.id == r.json()["check_id"]
    assert row.verdict == "LIKELY_SCAM" and row.engine == "fake"
    assert row.latency_ms >= 0 and row.truncated is False
    assert row.help_improve is False
    assert row.message_text is None and row.probability is None
    assert row.scam_type is None and row.channel is None
    # nothing from the message leaked into any column
    assert not any("555" in str(v) or "gmail" in str(v) for v in row.model_dump().values())


def test_opt_in_stores_masked_text(client, storage):
    r = client.post("/api/check", json={"text": MSG, "channel": "sms", "help_improve": True})
    (row,) = rows(storage, CheckRecord)
    assert row.help_improve is True
    assert row.message_text == mask_pii(MSG)
    assert "123-4567" not in row.message_text and "gmail" not in row.message_text
    assert row.probability == 0.9 and row.scam_type == "family_emergency" and row.channel == "sms"


def test_feedback_round_trip(client, storage):
    check_id = client.post("/api/check", json={"text": MSG}).json()["check_id"]
    r = client.post(f"/api/checks/{check_id}/feedback", json={"correct": False, "true_label": "safe"})
    assert r.status_code == 201
    assert r.json() == {"id": 1, "check_id": check_id, "correct": False, "true_label": "safe"}
    (fb,) = rows(storage, FeedbackRecord)
    assert fb.check_id == check_id and fb.correct is False and fb.true_label == "safe"


def test_feedback_without_label(client):
    check_id = client.post("/api/check", json={"text": MSG}).json()["check_id"]
    r = client.post(f"/api/checks/{check_id}/feedback", json={"correct": True})
    assert r.status_code == 201 and r.json()["true_label"] is None


def test_feedback_unknown_check_is_404(client):
    assert client.post("/api/checks/999/feedback", json={"correct": True}).status_code == 404


def test_feedback_rejects_bad_label(client):
    check_id = client.post("/api/check", json={"text": MSG}).json()["check_id"]
    r = client.post(f"/api/checks/{check_id}/feedback", json={"correct": False, "true_label": "maybe"})
    assert r.status_code == 422


def test_failed_check_is_not_logged(storage):
    c = TestClient(create_app(Settings(), None, storage))  # no key -> 503
    assert c.post("/api/check", json={"text": MSG}).status_code == 503
    assert rows(storage, CheckRecord) == []
