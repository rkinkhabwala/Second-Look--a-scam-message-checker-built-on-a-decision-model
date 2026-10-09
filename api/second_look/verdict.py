"""Turn engine answers into the user-facing result (spec §7).

Pure and deterministic: no model calls, no I/O. Every sentence a user sees is
one of the fixed templates below.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from second_look.config import Settings, get_settings
from second_look.engines.base import Answer
from second_look.questions import CHOICE_FLAG_NONE_OPTION, NOUL_FLAG_KEYS


class VerdictLabel(StrEnum):
    LIKELY_SCAM = "LIKELY_SCAM"
    UNSURE = "UNSURE"
    PROBABLY_SAFE = "PROBABLY_SAFE"


class Verdict(BaseModel):
    verdict: VerdictLabel
    probability: float
    scam_type: str | None
    red_flags: list[str]
    next_steps: list[str]


# ------------------------------------------------------------------ templates

NOUL_FLAG_TEMPLATES: dict[str, str] = {
    "urgency": "Pressures you to act right away or face a penalty — real organizations give you time to check.",
    "asks_credentials": "Asks for a password, PIN, one-time code, or account number — legitimate companies never ask for these by message.",
    "link_mismatch": "Contains a link to a website that doesn't belong to the organization the message claims to be from.",
    "too_good": "Offers money, prizes, or returns that are too good to be true.",
}

CHOICE_FLAG_TEMPLATES: dict[str, dict[str, str]] = {
    "payment_method": {
        "gift_card": "Asks for payment in gift cards — no real company or agency does this.",
        "crypto": "Asks for payment in cryptocurrency — agencies and real businesses don't demand crypto.",
        "wire_or_zelle": "Asks you to pay by wire transfer or an app like Zelle — money sent this way is almost impossible to get back.",
        "card_on_link": "Asks you to enter card details on a linked page — a common way to steal card numbers.",
    },
    "impersonation": {
        "bank": "Claims to be from a bank or card company — scammers often pretend to be banks.",
        "government_agency": "Claims to be from a government agency — agencies rarely make first contact by text or DM.",
        "delivery_company": "Claims to be from a delivery company — fake delivery notices are one of the most common scams.",
        "tech_company": "Claims to be from a tech company's support team — real support doesn't reach out to you first.",
        "employer_or_recruiter": "Claims to be from an employer or recruiter — real employers don't ask you to pay to get a job.",
        "family_member": "Claims to be a relative or friend — scammers pose as family to create panic.",
    },
}

SCAM_TYPE_STEPS: dict[str, list[str]] = {
    "delivery_fee": [
        "Track your package only on the carrier's official website or app — type the address yourself.",
        "Delivery companies don't ask for redelivery fees by text.",
    ],
    "toll_or_fine": [
        "Check your balance only on your state's official toll website — type the address yourself.",
        "For a ticket or fine, contact the court or agency using contact details from its official website.",
    ],
    "bank_or_card": [
        "Call your bank using the number on the back of your card or on its official website.",
        "If you shared card details, ask your bank to freeze the card right away.",
    ],
    "government_impersonation": [
        "Contact the agency using contact details from its official .gov website.",
        "Government agencies don't demand payment by gift card, crypto, or wire transfer.",
    ],
    "tech_support": [
        "Don't call the number in the message or let anyone access your device remotely.",
        "If you're worried, contact the company through its official website or app.",
    ],
    "job_offer": [
        "Research the company independently and apply only through its official careers page.",
        "Never pay a fee or buy equipment to start a job.",
    ],
    "romance": [
        "Never send money or gift cards to someone you haven't met in person.",
        "Talk to a friend or family member you trust about this relationship.",
    ],
    "investment_crypto": [
        "Don't send money or crypto to anyone who promises guaranteed or very high returns.",
        "Check whether the firm is registered with your country's financial regulator.",
    ],
    "prize_or_lottery": [
        "You can't win a contest you didn't enter, and real prizes never require a fee.",
    ],
    "account_verification": [
        "Sign in only by typing the company's address yourself or using its official app.",
        "If you entered a password, change it now and turn on two-factor authentication.",
    ],
    "family_emergency": [
        "Call the person on a number you already have before doing anything else.",
        "Ask a question only the real person would know the answer to.",
    ],
    "other_scam": [
        "Contact the organization using contact details you already trust, not ones in this message.",
    ],
    "not_a_scam": [],
}

HOW_TO_CHECK = (
    "We're not sure about this one. Contact the sender using a phone number or website you already trust — "
    "not the ones in this message."
)
SAFE_CAVEAT = (
    "This looks safe, but this check can be wrong. If the message asks for money or personal details, "
    "verify it through a channel you already trust."
)
ALWAYS_STEPS: list[str] = [
    "Don't click links in a message you're unsure about — type the website address yourself.",
    "Don't reply to a message you're unsure about — replying confirms your number is active.",
    "Report scams to the FTC at reportfraud.ftc.gov.",
    "Forward scam texts to 7726 (SPAM) to report them to your carrier.",
]


def all_templates() -> set[str]:
    """Every sentence verdict.py can output; used by tests to prove nothing is generated."""
    out = set(NOUL_FLAG_TEMPLATES.values()) | {HOW_TO_CHECK, SAFE_CAVEAT, *ALWAYS_STEPS}
    for table in CHOICE_FLAG_TEMPLATES.values():
        out |= set(table.values())
    for steps in SCAM_TYPE_STEPS.values():
        out |= set(steps)
    return out


# ---------------------------------------------------------------------- logic


def classify(p: float, risk_confidence: float | None, settings: Settings) -> VerdictLabel:
    """Abstain band: safe < p < scam, or a low-confidence risk_level, gives UNSURE.

    A missing risk_level confidence doesn't trigger abstention; the band alone decides.
    """
    if settings.safe_threshold < p < settings.scam_threshold:
        return VerdictLabel.UNSURE
    if risk_confidence is not None and risk_confidence < settings.min_risk_confidence:
        return VerdictLabel.UNSURE
    return VerdictLabel.LIKELY_SCAM if p >= settings.scam_threshold else VerdictLabel.PROBABLY_SAFE


def red_flags(answers: dict[str, Answer], threshold: float) -> list[str]:
    flags: list[str] = []
    for key in NOUL_FLAG_KEYS:
        a = answers.get(key)
        if a is not None and float(a.value) >= threshold:
            flags.append(NOUL_FLAG_TEMPLATES[key])
    for key, none_option in CHOICE_FLAG_NONE_OPTION.items():
        a = answers.get(key)
        if a is None:
            continue
        for option, template in CHOICE_FLAG_TEMPLATES[key].items():
            if option != none_option and a.probabilities.get(option, 0.0) >= threshold:
                flags.append(template)
    return flags


def next_steps(verdict: VerdictLabel, scam_type: str | None) -> list[str]:
    steps: list[str] = []
    if verdict is VerdictLabel.UNSURE:
        steps.append(HOW_TO_CHECK)
    if verdict is VerdictLabel.PROBABLY_SAFE:
        steps.append(SAFE_CAVEAT)
    if scam_type:
        steps += SCAM_TYPE_STEPS.get(scam_type, [])
    steps += ALWAYS_STEPS
    return list(dict.fromkeys(steps))


def make_verdict(answers: dict[str, Answer], settings: Settings | None = None) -> Verdict:
    settings = settings or get_settings()
    p = float(answers["is_scam"].value)
    risk = answers.get("risk_level")
    label = classify(p, risk.confidence if risk else None, settings)

    st = answers.get("scam_type")
    scam_type = str(st.value) if st is not None else None
    if label is VerdictLabel.PROBABLY_SAFE or scam_type == "not_a_scam":
        scam_type = None

    return Verdict(
        verdict=label,
        probability=p,
        scam_type=scam_type,
        red_flags=red_flags(answers, settings.red_flag_threshold),
        next_steps=next_steps(label, scam_type),
    )
