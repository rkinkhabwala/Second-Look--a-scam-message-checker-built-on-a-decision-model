"""The question set (spec §6): the single source of truth for what every engine is asked.

One atomic decision per question; Jev reads questions literally and compound
questions hurt accuracy. "None" options go last because Jev 1.13 is documented
to lean toward the first Choice option.
"""

from __future__ import annotations

from second_look.engines.base import Choice, Noul, Score

IS_SCAM = Noul(instructions="This message is a scam or fraud attempt.")

SCAM_TYPE = Choice(
    instructions="Which kind of scam this message is.",
    criteria={
        "delivery_fee": "Claims a package is held and asks for a small fee or address update.",
        "toll_or_fine": "Claims an unpaid toll, parking ticket, or traffic fine is due.",
        "bank_or_card": "Claims a problem with a bank account, card, or suspicious transaction.",
        "government_impersonation": "Pretends to be a tax office, court, police, or other government agency.",
        "tech_support": "Claims a device or account is infected or compromised and offers support.",
        "job_offer": "Offers a job, task-based work, or easy income that requires upfront steps.",
        "romance": "Builds a personal or romantic relationship in order to ask for money.",
        "investment_crypto": "Promotes a crypto, trading, or investment opportunity with high returns.",
        "prize_or_lottery": "Says the reader won a prize, lottery, or giveaway that must be claimed.",
        "account_verification": "Asks the reader to verify, unlock, or reactivate an online account.",
        "family_emergency": "Pretends to be a relative or friend in urgent trouble needing money.",
        "other_scam": "Is a scam that fits none of the other kinds.",
        "not_a_scam": "Is not a scam.",
    },
)

URGENCY = Noul(
    instructions="The message pressures the reader to act immediately or face a penalty."
)

PAYMENT_METHOD = Choice(
    instructions="Which payment method the message asks the reader to use.",
    criteria={
        "gift_card": "Asks for payment with gift cards or gift card codes.",
        "crypto": "Asks for payment in Bitcoin or another cryptocurrency.",
        "wire_or_zelle": "Asks for a wire transfer, Zelle, Venmo, Cash App, or similar instant transfer.",
        "card_on_link": "Asks the reader to enter card details on a linked page.",
        "none_requested": "Does not ask for any payment.",
    },
)

ASKS_CREDENTIALS = Noul(
    instructions="The message asks for a password, PIN, one-time code, or account number."
)

IMPERSONATION = Choice(
    instructions="Which kind of organization or person the message claims to be from.",
    criteria={
        "bank": "Claims to be a bank, card issuer, or payment service.",
        "government_agency": "Claims to be a government agency, court, or police.",
        "delivery_company": "Claims to be a postal or package delivery company.",
        "tech_company": "Claims to be a technology company or its support team.",
        "employer_or_recruiter": "Claims to be an employer, recruiter, or staffing agency.",
        "family_member": "Claims to be a relative or friend of the reader.",
        "none": "Does not claim to be from any of these.",
    },
)

LINK_MISMATCH = Noul(
    instructions="A link's domain does not match the organization the message claims to be from."
)

TOO_GOOD = Noul(
    instructions="The message offers money, prizes, or returns that are unrealistically generous."
)

RISK_LEVEL = Score(
    instructions="How risky this message is for the reader.",
    criteria=["safe", "probably safe", "suspicious", "likely scam", "scam"],
)

QUESTIONS: dict[str, Noul | Choice | Score] = {
    "is_scam": IS_SCAM,
    "scam_type": SCAM_TYPE,
    "urgency": URGENCY,
    "payment_method": PAYMENT_METHOD,
    "asks_credentials": ASKS_CREDENTIALS,
    "impersonation": IMPERSONATION,
    "link_mismatch": LINK_MISMATCH,
    "too_good": TOO_GOOD,
    "risk_level": RISK_LEVEL,
}

# Questions that can produce red flags. Noul flags fire on P(true); Choice flags
# fire on any option other than the listed "none" option.
NOUL_FLAG_KEYS: tuple[str, ...] = ("urgency", "asks_credentials", "link_mismatch", "too_good")
CHOICE_FLAG_NONE_OPTION: dict[str, str] = {
    "payment_method": "none_requested",
    "impersonation": "none",
}
