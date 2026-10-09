import ast
import inspect
from dataclasses import replace

import pytest

from second_look import verdict as verdict_module
from second_look.config import Settings
from second_look.engines.fake import FakeEngine, choice, noul, score
from second_look.questions import QUESTIONS
from second_look.verdict import (
    ALWAYS_STEPS,
    CHOICE_FLAG_TEMPLATES,
    HOW_TO_CHECK,
    NOUL_FLAG_TEMPLATES,
    SAFE_CAVEAT,
    SCAM_TYPE_STEPS,
    VerdictLabel,
    all_templates,
    make_verdict,
)

S = Settings()


def answers(
    p=0.9,
    *,
    scam_type="delivery_fee",
    risk_conf=0.8,
    flags=None,
    payment=None,
    impersonation=None,
):
    """A full answer set; every flag below threshold unless overridden."""
    flags = flags or {}
    st_probs = {o: 0.0 for o in QUESTIONS["scam_type"].criteria}
    st_probs[scam_type] = 1.0
    pay = {o: 0.0 for o in QUESTIONS["payment_method"].criteria}
    pay.update(payment or {"none_requested": 1.0})
    imp = {o: 0.0 for o in QUESTIONS["impersonation"].criteria}
    imp.update(impersonation or {"none": 1.0})
    return {
        "is_scam": noul(p),
        "scam_type": choice(st_probs),
        "urgency": noul(flags.get("urgency", 0.1)),
        "payment_method": choice(pay),
        "asks_credentials": noul(flags.get("asks_credentials", 0.1)),
        "impersonation": choice(imp),
        "link_mismatch": noul(flags.get("link_mismatch", 0.1)),
        "too_good": noul(flags.get("too_good", 0.1)),
        "risk_level": score([0.0, 0.0, 0.1, 0.2, 0.7], confidence=risk_conf),
    }


async def verdict_via_fake_engine(ans, settings=S):
    engine = FakeEngine(ans)
    return make_verdict(await engine.decide({"message": "x"}, QUESTIONS), settings)


# -------------------------------------------------------------- verdict bands


@pytest.mark.parametrize(
    "p, expected",
    [
        (0.0, VerdictLabel.PROBABLY_SAFE),
        (0.25, VerdictLabel.PROBABLY_SAFE),  # boundary: p <= 0.25 is safe
        (0.2501, VerdictLabel.UNSURE),
        (0.5, VerdictLabel.UNSURE),
        (0.7499, VerdictLabel.UNSURE),
        (0.75, VerdictLabel.LIKELY_SCAM),  # boundary: p >= 0.75 is scam
        (1.0, VerdictLabel.LIKELY_SCAM),
    ],
)
async def test_probability_bands(p, expected):
    v = await verdict_via_fake_engine(answers(p))
    assert v.verdict is expected
    assert v.probability == p


@pytest.mark.parametrize("p", [0.95, 0.05])
async def test_low_risk_confidence_abstains_even_when_p_is_extreme(p):
    v = await verdict_via_fake_engine(answers(p, risk_conf=0.39))
    assert v.verdict is VerdictLabel.UNSURE


async def test_risk_confidence_exactly_at_threshold_does_not_abstain():
    assert (await verdict_via_fake_engine(answers(0.9, risk_conf=0.4))).verdict is VerdictLabel.LIKELY_SCAM


async def test_missing_risk_confidence_uses_band_only():
    ans = answers(0.9)
    ans["risk_level"] = ans["risk_level"].model_copy(update={"confidence": None})
    assert (await verdict_via_fake_engine(ans)).verdict is VerdictLabel.LIKELY_SCAM


async def test_thresholds_come_from_config():
    strict = replace(S, scam_threshold=0.95, safe_threshold=0.05, min_risk_confidence=0.9)
    assert (await verdict_via_fake_engine(answers(0.9), strict)).verdict is VerdictLabel.UNSURE
    assert (await verdict_via_fake_engine(answers(0.9, risk_conf=0.95), strict)).verdict is VerdictLabel.UNSURE
    assert (await verdict_via_fake_engine(answers(0.96, risk_conf=0.95), strict)).verdict is VerdictLabel.LIKELY_SCAM


# ------------------------------------------------------------------ red flags


@pytest.mark.parametrize("key", list(NOUL_FLAG_TEMPLATES))
async def test_noul_flag_threshold_boundary(key):
    shown = await verdict_via_fake_engine(answers(flags={key: 0.6}))
    hidden = await verdict_via_fake_engine(answers(flags={key: 0.5999}))
    assert shown.red_flags == [NOUL_FLAG_TEMPLATES[key]]
    assert hidden.red_flags == []


@pytest.mark.parametrize(
    "question, option",
    [(q, o) for q, table in CHOICE_FLAG_TEMPLATES.items() for o in table],
)
async def test_choice_flags(question, option):
    none_opt = {"payment_method": "none_requested", "impersonation": "none"}[question]
    kw = "payment" if question == "payment_method" else "impersonation"
    shown = await verdict_via_fake_engine(answers(**{kw: {option: 0.6, none_opt: 0.4}}))
    hidden = await verdict_via_fake_engine(answers(**{kw: {option: 0.59, none_opt: 0.41}}))
    assert shown.red_flags == [CHOICE_FLAG_TEMPLATES[question][option]]
    assert hidden.red_flags == []


async def test_none_options_never_flag():
    v = await verdict_via_fake_engine(answers())  # none_requested / none at 1.0
    assert v.red_flags == []


async def test_spec_example_gift_card_template():
    v = await verdict_via_fake_engine(answers(payment={"gift_card": 0.9, "none_requested": 0.1}))
    assert "Asks for payment in gift cards — no real company or agency does this." in v.red_flags


async def test_multiple_flags_in_stable_order():
    v = await verdict_via_fake_engine(answers(
        flags={"urgency": 0.9, "too_good": 0.8},
        payment={"crypto": 0.7, "none_requested": 0.3},
        impersonation={"bank": 0.65, "none": 0.35},
    ))
    assert v.red_flags == [
        NOUL_FLAG_TEMPLATES["urgency"],
        NOUL_FLAG_TEMPLATES["too_good"],
        CHOICE_FLAG_TEMPLATES["payment_method"]["crypto"],
        CHOICE_FLAG_TEMPLATES["impersonation"]["bank"],
    ]


# ----------------------------------------------------------------- next steps


@pytest.mark.parametrize("scam_type", [t for t in QUESTIONS["scam_type"].criteria if t != "not_a_scam"])
async def test_every_scam_type_has_specific_steps(scam_type):
    v = await verdict_via_fake_engine(answers(0.9, scam_type=scam_type))
    assert v.scam_type == scam_type
    assert SCAM_TYPE_STEPS[scam_type]
    assert v.next_steps[: len(SCAM_TYPE_STEPS[scam_type])] == SCAM_TYPE_STEPS[scam_type]


def test_step_table_covers_exactly_the_question_options():
    assert set(SCAM_TYPE_STEPS) == set(QUESTIONS["scam_type"].criteria)


@pytest.mark.parametrize("p", [0.95, 0.5, 0.05])
async def test_reporting_steps_always_included(p):
    v = await verdict_via_fake_engine(answers(p))
    assert v.next_steps[-len(ALWAYS_STEPS):] == ALWAYS_STEPS
    joined = " ".join(v.next_steps)
    assert "reportfraud.ftc.gov" in joined and "7726" in joined


async def test_probably_safe_says_it_can_be_wrong():
    v = await verdict_via_fake_engine(answers(0.1, scam_type="not_a_scam"))
    assert v.verdict is VerdictLabel.PROBABLY_SAFE
    assert v.next_steps[0] == SAFE_CAVEAT
    assert v.scam_type is None


async def test_probably_safe_hides_scam_type_even_if_engine_named_one():
    v = await verdict_via_fake_engine(answers(0.1, scam_type="toll_or_fine"))
    assert v.scam_type is None
    assert not set(SCAM_TYPE_STEPS["toll_or_fine"]) & set(v.next_steps)


async def test_unsure_explains_how_to_check():
    v = await verdict_via_fake_engine(answers(0.5, scam_type="family_emergency"))
    assert v.next_steps[0] == HOW_TO_CHECK
    assert "Call the person on a number you already have before doing anything else." in v.next_steps


async def test_unsure_with_not_a_scam_type_has_no_scam_type():
    v = await verdict_via_fake_engine(answers(0.5, scam_type="not_a_scam"))
    assert v.scam_type is None and v.next_steps[0] == HOW_TO_CHECK


# ------------------------------------------------------------- no generated text


@pytest.mark.parametrize("p", [0.95, 0.5, 0.05])
async def test_every_sentence_is_a_template(p):
    v = await verdict_via_fake_engine(answers(
        p, flags={k: 0.99 for k in NOUL_FLAG_TEMPLATES},
        payment={"gift_card": 0.9, "none_requested": 0.1},
        impersonation={"government_agency": 0.9, "none": 0.1},
    ))
    templates = all_templates()
    for sentence in v.red_flags + v.next_steps:
        assert sentence in templates


def test_verdict_module_imports_no_engine_implementations():
    tree = ast.parse(inspect.getsource(verdict_module))
    imported = {
        n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    } | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.startswith("second_look.engines.") and m != "second_look.engines.base" for m in imported)
    assert not {"httpx", "transformers"} & imported
