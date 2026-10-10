import math

import pytest

from second_look.engines.base import Choice, Noul, Score
from second_look.engines.hf_scorer import common_prefix_len
from second_look.engines.llm_scoring import (
    PROMPT_VERSION,
    LLMScoringEngine,
    build_prompts,
    question_text,
    softmax,
    to_answer,
)
from second_look.engines.render import render_state
from second_look.preprocess import build_state
from second_look.questions import QUESTIONS


class FakeScorer:
    model_id = "fake/model@deadbeef"

    def __init__(self, fn=None):
        self.calls = []
        self.fn = fn or (lambda i, k: [float(k - j) for j in range(k)])  # favours option A

    def letter_logits(self, system, context, suffixes, n_options):
        self.calls.append((system, context, suffixes, n_options))
        return [self.fn(i, k) for i, k in enumerate(n_options)]


@pytest.fixture
def state():
    return build_state("USPS: parcel held. Pay $1.99 at https://usps.com.track-fee.top", claimed_sender="USPS")[0]


# ---------------------------------------------------------------- rendering


def test_render_state_includes_extracted_features(state):
    text = render_state(state)
    assert text.startswith("Message (text message):\nUSPS: parcel held.")
    assert "Claimed sender: USPS" in text
    assert "registered domain track-fee.top" in text
    assert "Money amounts: $1.99" in text
    assert "Phone numbers: none" in text


def test_render_marks_shorteners_and_punycode():
    s, _ = build_state("go to bit.ly/x or https://xn--pypal-4ve.com/a")
    text = render_state(s)
    assert "link shortener" in text and "displays as pаypal.com" in text


def test_render_plain_string_state():
    assert render_state("hello") == "Message:\nhello"


# ------------------------------------------------------------------ prompts


def test_every_question_gets_one_letter_per_option():
    for p in build_prompts(QUESTIONS):
        letters = [line[0] for line in p.text.splitlines() if len(line) > 2 and line[1:3] == ". "]
        assert letters == [chr(65 + i) for i in range(len(p.options))], p.key


def test_noul_prompt_is_true_false():
    text = question_text(Noul(instructions="The message is urgent."))
    assert "Statement: The message is urgent." in text and "A. True" in text and "B. False" in text


def test_choice_and_score_prompts_keep_order():
    c = question_text(QUESTIONS["scam_type"])
    assert c.index("A. Claims a package") < c.index("M. Is not a scam.")
    s = question_text(QUESTIONS["risk_level"])
    assert "lowest to highest" in s and s.index("A. safe") < s.index("E. scam")


def test_too_many_options_rejected():
    with pytest.raises(ValueError):
        build_prompts({"big": Choice(instructions="x", criteria={str(i): "d" for i in range(27)})})


# ---------------------------------------------------------- probabilities


def test_softmax_is_stable_and_normalized():
    assert softmax([1000.0, 1000.0]) == [0.5, 0.5]
    assert sum(softmax([1.0, 2.0, 3.0])) == pytest.approx(1.0)


def test_noul_answer():
    (p,) = build_prompts({"q": Noul(instructions="x")})
    a = to_answer(p, [math.log(3), 0.0])  # P(true) = 0.75
    assert a.type == "noul" and a.value == pytest.approx(0.75)
    assert a.probabilities == {"true": pytest.approx(0.75), "false": pytest.approx(0.25)}
    assert a.confidence == pytest.approx(0.5)
    assert a.raw == {"letter_logits": [math.log(3), 0.0]}


def test_choice_answer_and_confidence():
    (p,) = build_prompts({"q": Choice(instructions="x", criteria={"a": "A", "b": "B", "c": "C"})})
    a = to_answer(p, [math.log(0.2), math.log(0.7), math.log(0.1)])
    assert a.value == "b"
    assert a.confidence == pytest.approx(0.5)


def test_score_expected_level_matches_jev_definition():
    (p,) = build_prompts({"q": Score(instructions="x", criteria=["lo", "mid", "hi"])})
    a = to_answer(p, [math.log(1e-9), math.log(0.57), math.log(0.43)])
    assert a.value == pytest.approx(1.43, abs=1e-6)
    assert set(a.probabilities) == {"0", "1", "2"}


# ------------------------------------------------------------------- engine


async def test_engine_scores_all_questions_in_one_scorer_call(state):
    scorer = FakeScorer()
    engine = LLMScoringEngine(scorer)
    result = await engine.run(state, QUESTIONS)
    assert len(scorer.calls) == 1
    system, context, suffixes, n_options = scorer.calls[0]
    assert context == render_state(state)
    assert len(suffixes) == 9 and n_options[1] == 13
    assert set(result.answers) == set(QUESTIONS)
    assert result.answers["scam_type"].value == "delivery_fee"  # fake favours A
    assert result.model == "fake/model@deadbeef" and result.truncated is False


async def test_decide_matches_protocol(state):
    answers = await LLMScoringEngine(FakeScorer()).decide(state, QUESTIONS)
    assert answers["is_scam"].type == "noul"


async def test_long_message_is_truncated_and_reported():
    long_state, _ = build_state("START " + "word " * 20_000 + " END")
    scorer = FakeScorer()
    result = await LLMScoringEngine(scorer, max_message_tokens=500).run(long_state, QUESTIONS)
    assert result.truncated
    context = scorer.calls[0][1]
    assert "START" in context and "END" in context and "characters removed" in context


def test_payload_changes_with_model_and_prompt_version(state):
    a, _ = LLMScoringEngine(FakeScorer()).build_payload(state, QUESTIONS)
    other = FakeScorer()
    other.model_id = "fake/other@cafebabe"
    b, _ = LLMScoringEngine(other).build_payload(state, QUESTIONS)
    assert a["prompt_version"] == PROMPT_VERSION
    assert a != b and a["context"] == b["context"]


# --------------------------------------------------------------- HF helper


def test_common_prefix_len():
    assert common_prefix_len([[1, 2, 3, 4], [1, 2, 5], [1, 2, 3]]) == 2
    assert common_prefix_len([[1, 2, 3], [1, 2, 3]]) == 2  # always leaves one token per row
    assert common_prefix_len([[1], [1]]) == 0
