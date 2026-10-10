import math

import numpy as np
import pytest

from second_look.calibration import (
    CalibratedEngine,
    Calibration,
    answer_logits,
    apply_temperature,
    binary_logit,
    fit_calibration,
    fit_temperature_binary,
    fit_temperature_multiclass,
)
from second_look.engines.base import EngineResult
from second_look.engines.llm_scoring import build_prompts, softmax
from second_look.engines.llm_scoring import to_answer as llm_answer
from second_look.engines.nli import sigmoid
from second_look.engines.nli import to_answer as nli_answer
from second_look.eval.metrics import ece
from second_look.questions import QUESTIONS

PROMPTS = {p.key: p for p in build_prompts(QUESTIONS)}


def synthetic_binary(true_t, n=4000, seed=0):
    """Logits whose correct temperature is true_t: labels drawn from sigmoid(z / true_t)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 4, n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-z / true_t))).astype(int)
    return z, y


@pytest.mark.parametrize("true_t", [0.5, 1.0, 3.0])
def test_fit_binary_recovers_temperature(true_t):
    z, y = synthetic_binary(true_t)
    assert fit_temperature_binary(z, y) == pytest.approx(true_t, rel=0.15)


def test_fit_multiclass_recovers_temperature():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 3, (4000, 4))
    probs = np.exp(logits / 2.5) / np.exp(logits / 2.5).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(4, p=p) for p in probs])
    assert fit_temperature_multiclass(logits, y) == pytest.approx(2.5, rel=0.15)


def test_calibration_lowers_ece_on_overconfident_data():
    z, y = synthetic_binary(3.0)  # raw sigmoid(z) is overconfident by a factor of 3
    t = fit_temperature_binary(z, y)
    raw = 1 / (1 + np.exp(-z))
    cal = 1 / (1 + np.exp(-z / t))
    assert ece(y, cal) < ece(y, raw) / 2


def test_binary_logit_for_llm_and_nli():
    assert binary_logit([2.0, -1.0]) == 3.0
    assert binary_logit([1.5]) == 1.5


def test_apply_temperature_llm_noul():
    a = llm_answer(PROMPTS["is_scam"], [4.0, 0.0])
    cal = apply_temperature(a, QUESTIONS["is_scam"], 2.0)
    assert cal.value == pytest.approx(sigmoid(2.0))
    assert cal.raw["temperature"] == 2.0 and cal.raw["letter_logits"] == [4.0, 0.0]
    assert apply_temperature(a, QUESTIONS["is_scam"], 1.0) is a


def test_apply_temperature_nli_noul():
    a = nli_answer(QUESTIONS["urgency"], ["true"], [3.0])
    assert apply_temperature(a, QUESTIONS["urgency"], 3.0).value == pytest.approx(0.5 + 0.5 * math.tanh(0.5))


def test_apply_temperature_choice_and_score_keep_option_keys():
    logits = [3.0] + [0.0] * 12
    a = llm_answer(PROMPTS["scam_type"], logits)
    cal = apply_temperature(a, QUESTIONS["scam_type"], 4.0)
    assert list(cal.probabilities) == list(QUESTIONS["scam_type"].criteria)
    assert cal.probabilities["delivery_fee"] == pytest.approx(softmax([x / 4 for x in logits])[0])
    assert cal.confidence < a.confidence
    s = apply_temperature(llm_answer(PROMPTS["risk_level"], [0, 0, 0, 0, 5.0]), QUESTIONS["risk_level"], 5.0)
    assert 2.0 < s.value < 4.0


def test_jev_answers_pass_through():
    from second_look.engines.fake import noul

    a = noul(0.9)
    assert answer_logits(a) is None
    assert apply_temperature(a, QUESTIONS["is_scam"], 2.0) is a


def test_fit_calibration_only_fits_labeled_questions():
    z, y = synthetic_binary(2.0, n=200)
    answers = [{"is_scam": llm_answer(PROMPTS["is_scam"], [zi, 0.0]),
                "urgency": llm_answer(PROMPTS["urgency"], [zi, 0.0])} for zi in z]
    cal = fit_calibration("llm", "m@1", "llm-v1", answers, {"is_scam": list(y), "scam_type": [None] * 200}, QUESTIONS)
    assert set(cal.temperatures) == {"is_scam"}
    assert cal.temperatures["is_scam"] == pytest.approx(2.0, rel=0.4)
    assert cal.fit_on == {"examples_per_question": {"is_scam": 200}}


def test_fit_calibration_skips_questions_with_too_few_labels():
    answers = [{"scam_type": llm_answer(PROMPTS["scam_type"], [1.0] + [0.0] * 12)}] * 10
    cal = fit_calibration("llm", "m", None, answers, {"scam_type": ["delivery_fee"] * 10}, QUESTIONS)
    assert cal.temperatures == {}


def test_save_load_round_trip(tmp_path):
    cal = Calibration("nli", "x/y@12345678", "nli-v1", {"is_scam": 1.7}, {"examples_per_question": {"is_scam": 1000}})
    path = cal.save(tmp_path)
    assert path.name == "nli.json"
    assert Calibration.load("nli", tmp_path) == cal
    assert Calibration.load("llm", tmp_path) is None


class _Inner:
    name = "llm"
    model = "m@1"

    async def run(self, state, questions):
        return EngineResult(answers={"is_scam": llm_answer(PROMPTS["is_scam"], [4.0, 0.0])}, model=self.model)

    def build_payload(self, state, questions):
        return {"p": 1}, False


async def test_calibrated_engine_wraps_and_applies():
    eng = CalibratedEngine(_Inner(), Calibration("llm", "m@1", None, {"is_scam": 2.0}))
    answers = await eng.decide({}, {"is_scam": QUESTIONS["is_scam"]})
    assert answers["is_scam"].value == pytest.approx(sigmoid(2.0))
    assert eng.build_payload({}, {}) == ({"p": 1}, False)  # caches stay raw


def test_calibrated_engine_refuses_mismatched_model():
    with pytest.raises(ValueError):
        CalibratedEngine(_Inner(), Calibration("llm", "other@2", None, {"is_scam": 2.0}))


def test_registry_wraps_only_matching_calibration(monkeypatch, tmp_path):
    from second_look import calibration
    from second_look.engines.registry import with_calibration

    monkeypatch.setattr(calibration, "CALIBRATION_DIR", tmp_path)
    inner = _Inner()
    assert with_calibration(inner) is inner  # no file
    Calibration("llm", "m@1", None, {"is_scam": 2.0}).save(tmp_path)
    assert isinstance(with_calibration(inner), CalibratedEngine)
    Calibration("llm", "different@9", None, {"is_scam": 2.0}).save(tmp_path)
    assert with_calibration(inner) is inner  # fit on another model: ignored
