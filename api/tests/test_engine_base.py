import pytest
from pydantic import ValidationError

from second_look.engines import Answer, Choice, DecisionEngine, Noul, Score
from second_look.engines.fake import FakeEngine, choice, noul, score


def test_question_models_default_types():
    assert Noul(instructions="x").type == "noul"
    assert Choice(instructions="x", criteria={"a": "A"}).type == "choice"
    assert Score(instructions="x", criteria=["lo", "hi"]).type == "score"


def test_question_models_reject_wrong_type():
    with pytest.raises(ValidationError):
        Noul(type="choice", instructions="x")
    with pytest.raises(ValidationError):
        Choice(instructions="x", criteria=["not", "a", "dict"])
    with pytest.raises(ValidationError):
        Score(instructions="x", criteria={"not": "a list"})


def test_answer_accepts_str_or_float_value():
    assert Answer(type="choice", value="a", probabilities={"a": 1.0}, confidence=1.0).value == "a"
    assert Answer(type="noul", value=0.3, probabilities={}, confidence=None).raw is None


def test_helpers():
    assert noul(0.8).probabilities == {"true": 0.8, "false": pytest.approx(0.2)}
    c = choice({"a": 0.7, "b": 0.2, "c": 0.1})
    assert c.value == "a" and c.confidence == pytest.approx(0.5)
    s = score([0.0, 0.57, 0.43])  # the worked example from Jev's score docs
    assert s.value == pytest.approx(1.43)
    assert s.probabilities == {"0": 0.0, "1": 0.57, "2": 0.43}


async def test_fake_engine_satisfies_protocol_and_returns_script():
    qs = {"is_scam": Noul(instructions="x"), "other": Noul(instructions="y")}
    engine = FakeEngine({"is_scam": noul(0.9), "unasked": noul(0.1)})
    assert isinstance(engine, DecisionEngine)
    out = await engine.decide({"message": "hi"}, qs)
    assert out == {"is_scam": noul(0.9)}  # only asked questions; missing ones omitted
    assert engine.calls == [({"message": "hi"}, qs)]


async def test_fake_engine_callable_script():
    engine = FakeEngine(lambda state, qs: {k: noul(0.5) for k in qs})
    out = await engine.decide("s", {"a": Noul(instructions="x")})
    assert out["a"].value == 0.5
