import pytest

from second_look.engines.nli import NLIEngine, middle_truncate_chars, sigmoid, to_answer
from second_look.engines.render import render_state
from second_look.preprocess import build_state
from second_look.questions import QUESTIONS, nli_hypotheses


class FakeNLI:
    model_id = "fake/nli@deadbeef"

    def __init__(self, fn=None):
        self.calls = []
        self.fn = fn or (lambda h: 2.0 if "scam" in h.lower() else -2.0)

    def entailment_scores(self, premise, hypotheses):
        self.calls.append((premise, hypotheses))
        return [self.fn(h) for h in hypotheses]


def test_every_option_has_a_standalone_hypothesis():
    for key, q in QUESTIONS.items():
        hyps = nli_hypotheses(key)
        expected = ["true"] if q.type == "noul" else (
            list(q.criteria) if q.type == "choice" else [str(i) for i in range(len(q.criteria))])
        assert list(hyps) == expected, key
        for h in hyps.values():
            assert h.endswith(".") and (h.startswith("This message") or h.startswith("The message")
                                        or h.startswith("A link")), h
            assert "any of these" not in h and "none of the other" not in h, h


def test_hypotheses_read_naturally():
    st = nli_hypotheses("scam_type")
    assert st["delivery_fee"] == "This message claims a package is held and asks for a small fee or address update."
    assert st["not_a_scam"] == "This message is not a scam."
    assert nli_hypotheses("risk_level")["3"] == "This message is likely a scam."


def test_sigmoid_symmetric_and_stable():
    assert sigmoid(0) == 0.5
    assert sigmoid(800) == 1.0 and sigmoid(-800) == pytest.approx(0.0)
    assert sigmoid(2) + sigmoid(-2) == pytest.approx(1.0)


def test_noul_answer():
    a = to_answer(QUESTIONS["urgency"], ["true"], [0.0])
    assert a.value == 0.5 and a.confidence == 0.0
    assert to_answer(QUESTIONS["urgency"], ["true"], [3.0]).value == pytest.approx(sigmoid(3.0))


def test_choice_and_score_answers():
    opts = list(QUESTIONS["payment_method"].criteria)
    a = to_answer(QUESTIONS["payment_method"], opts, [5.0, 0.0, 0.0, 0.0, 0.0])
    assert a.value == "gift_card" and sum(a.probabilities.values()) == pytest.approx(1.0)
    s = to_answer(QUESTIONS["risk_level"], [str(i) for i in range(5)], [-9.0, -9.0, -9.0, -9.0, 9.0])
    assert s.value == pytest.approx(4.0, abs=1e-3)


def test_middle_truncate_chars():
    assert middle_truncate_chars("short", 100) == ("short", False)
    out, t = middle_truncate_chars("A" * 50 + "B" * 50, 20)
    assert t and out.startswith("A" * 10) and out.endswith("B" * 10)


async def test_engine_scores_all_pairs_in_one_call():
    state, _ = build_state("You WON a $500 prize! Claim at bit.ly/x")
    scorer = FakeNLI()
    result = await NLIEngine(scorer).run(state, QUESTIONS)
    assert len(scorer.calls) == 1
    premise, hyps = scorer.calls[0]
    assert premise == render_state(state)
    assert len(hyps) == 5 + 13 + 5 + 7 + 5  # 5 noul + scam_type + payment + impersonation + risk levels
    assert set(result.answers) == set(QUESTIONS)
    assert result.answers["is_scam"].value == pytest.approx(sigmoid(2.0))
    assert result.model == "fake/nli@deadbeef"


async def test_long_message_truncated_and_reported():
    state, _ = build_state("HEAD " + "x" * 5_000 + " TAIL")
    scorer = FakeNLI()
    result = await NLIEngine(scorer).run(state, QUESTIONS)
    assert result.truncated
    assert "HEAD" in scorer.calls[0][0] and "TAIL" in scorer.calls[0][0]


def test_payload_includes_model_and_prompt_version():
    state, _ = build_state("hi")
    payload, truncated = NLIEngine(FakeNLI()).build_payload(state, QUESTIONS)
    assert payload["model"] == "fake/nli@deadbeef" and payload["prompt_version"] == "nli-v2"
    assert not truncated
