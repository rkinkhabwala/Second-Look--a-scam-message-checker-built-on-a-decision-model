"""SQLite storage for checks and feedback (SQLModel).

Privacy (spec §8): by default only the verdict, engine, latency and truncation
flag are logged. Message text (plus probability, scam type and channel) is kept
only when the user opts in with "help improve", and phone numbers and email
addresses in that text are masked before it is written.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy.pool import StaticPool
from sqlmodel import Field, Session, SQLModel, create_engine

from second_look.preprocess import phone_matches

TrueLabel = Literal["scam", "safe"]

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def mask_pii(text: str) -> str:
    """Replace email addresses with [email] and phone numbers with [phone].

    SMS short codes (e.g. 7726) aren't personal data and are left as-is.
    """
    text = _EMAIL.sub("[email]", text)
    spans = sorted((span for _, span, short in phone_matches(text) if not short), reverse=True)
    for start, end in spans:
        text = text[:start] + "[phone]" + text[end:]
    return text


def _now() -> datetime:
    return datetime.now(UTC)


class CheckRecord(SQLModel, table=True):
    __tablename__ = "checks"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=_now)
    verdict: str
    engine: str
    latency_ms: float
    truncated: bool
    help_improve: bool = False
    # Only filled when help_improve is true:
    message_text: str | None = None
    probability: float | None = None
    scam_type: str | None = None
    channel: str | None = None


class FeedbackRecord(SQLModel, table=True):
    __tablename__ = "feedback"

    id: int | None = Field(default=None, primary_key=True)
    check_id: int = Field(foreign_key="checks.id", index=True)
    created_at: datetime = Field(default_factory=_now)
    correct: bool
    true_label: str | None = None


class Storage:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self._engine = None  # created on first use so importing the app touches no disk

    @property
    def engine(self):
        if self._engine is None:
            kwargs: dict = {"connect_args": {"check_same_thread": False}}
            if self.db_url in ("sqlite://", "sqlite:///:memory:"):
                kwargs["poolclass"] = StaticPool
            self._engine = create_engine(self.db_url, **kwargs)
            SQLModel.metadata.create_all(self._engine)
        return self._engine

    def log_check(
        self,
        *,
        verdict: str,
        engine: str,
        latency_ms: float,
        truncated: bool,
        help_improve: bool = False,
        text: str | None = None,
        probability: float | None = None,
        scam_type: str | None = None,
        channel: str | None = None,
    ) -> int:
        record = CheckRecord(
            verdict=verdict, engine=engine, latency_ms=latency_ms, truncated=truncated,
            help_improve=help_improve,
        )
        if help_improve:
            record.message_text = mask_pii(text) if text else None
            record.probability = probability
            record.scam_type = scam_type
            record.channel = channel
        with Session(self.engine) as s:
            s.add(record)
            s.commit()
            s.refresh(record)
            return record.id

    def get_check(self, check_id: int) -> CheckRecord | None:
        with Session(self.engine) as s:
            return s.get(CheckRecord, check_id)

    def add_feedback(
        self, check_id: int, correct: bool, true_label: TrueLabel | None = None
    ) -> FeedbackRecord | None:
        """Returns None if the check doesn't exist."""
        with Session(self.engine) as s:
            if s.get(CheckRecord, check_id) is None:
                return None
            fb = FeedbackRecord(check_id=check_id, correct=correct, true_label=true_label)
            s.add(fb)
            s.commit()
            s.refresh(fb)
            return fb
