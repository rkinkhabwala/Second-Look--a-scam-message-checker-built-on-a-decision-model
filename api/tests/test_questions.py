from second_look.engines import Choice, Noul, Score
from second_look.questions import CHOICE_FLAG_NONE_OPTION, NOUL_FLAG_KEYS, QUESTIONS


def test_keys_and_types_match_spec():
    expected = {
        "is_scam": Noul,
        "scam_type": Choice,
        "urgency": Noul,
        "payment_method": Choice,
        "asks_credentials": Noul,
        "impersonation": Choice,
        "link_mismatch": Noul,
        "too_good": Noul,
        "risk_level": Score,
    }
    assert list(QUESTIONS) == list(expected)
    for key, cls in expected.items():
        assert type(QUESTIONS[key]) is cls, key


def test_noul_instructions_verbatim_from_spec():
    assert QUESTIONS["is_scam"].instructions == "This message is a scam or fraud attempt."
    assert QUESTIONS["urgency"].instructions == (
        "The message pressures the reader to act immediately or face a penalty."
    )
    assert QUESTIONS["asks_credentials"].instructions == (
        "The message asks for a password, PIN, one-time code, or account number."
    )
    assert QUESTIONS["link_mismatch"].instructions == (
        "A link's domain does not match the organization the message claims to be from."
    )
    assert QUESTIONS["too_good"].instructions == (
        "The message offers money, prizes, or returns that are unrealistically generous."
    )


def test_choice_options_match_spec():
    assert set(QUESTIONS["scam_type"].criteria) == {
        "delivery_fee", "toll_or_fine", "bank_or_card", "government_impersonation",
        "tech_support", "job_offer", "romance", "investment_crypto", "prize_or_lottery",
        "account_verification", "family_emergency", "other_scam", "not_a_scam",
    }
    assert set(QUESTIONS["payment_method"].criteria) == {
        "gift_card", "crypto", "wire_or_zelle", "card_on_link", "none_requested",
    }
    assert set(QUESTIONS["impersonation"].criteria) == {
        "bank", "government_agency", "delivery_company", "tech_company",
        "employer_or_recruiter", "family_member", "none",
    }


def test_none_options_are_last():
    # Jev 1.13 leans toward the first Choice option, so the "no" answer must not sit first.
    assert list(QUESTIONS["scam_type"].criteria)[-1] == "not_a_scam"
    assert list(QUESTIONS["payment_method"].criteria)[-1] == "none_requested"
    assert list(QUESTIONS["impersonation"].criteria)[-1] == "none"


def test_risk_level_order():
    assert QUESTIONS["risk_level"].criteria == [
        "safe", "probably safe", "suspicious", "likely scam", "scam",
    ]


def test_every_description_is_one_sentence():
    for key, q in QUESTIONS.items():
        texts = [q.instructions]
        if isinstance(q, Choice):
            texts += list(q.criteria.values())
        for t in texts:
            assert t.strip() and t.count(". ") == 0 and t.endswith("."), (key, t)


def test_flag_keys_reference_real_questions():
    for k in NOUL_FLAG_KEYS:
        assert isinstance(QUESTIONS[k], Noul)
    for k, none_opt in CHOICE_FLAG_NONE_OPTION.items():
        assert none_opt in QUESTIONS[k].criteria
