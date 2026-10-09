"""Runtime settings, read from environment variables with typed defaults.

Thresholds live here (not in verdict.py) so they can be tuned from the eval's
coverage/accuracy curve without touching logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

API_ROOT = Path(__file__).resolve().parent.parent


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # Verdict thresholds (spec §7)
    scam_threshold: float = 0.75
    safe_threshold: float = 0.25
    min_risk_confidence: float = 0.4
    red_flag_threshold: float = 0.6

    # Jev (spec §3, §5)
    typesafe_api_key: str | None = None
    jev_base_url: str = "https://api.typesafe.ai"
    jev_model: str = "jev-1.13.0"
    jev_timeout_s: float = 5.0  # per attempt
    jev_max_retries: int = 3
    jev_max_request_tokens: int = 64_000
    jev_max_state_tokens: int = 32_000  # state + longest question
    jev_price_per_mtok_usd: float = 0.042  # input tokens only; output is free (docs.typesafe.ai/models.md)

    # Storage
    db_url: str = field(default_factory=lambda: f"sqlite:///{API_ROOT / 'second_look.db'}")


def get_settings() -> Settings:
    d = Settings()
    return Settings(
        scam_threshold=_float("SL_SCAM_THRESHOLD", d.scam_threshold),
        safe_threshold=_float("SL_SAFE_THRESHOLD", d.safe_threshold),
        min_risk_confidence=_float("SL_MIN_RISK_CONFIDENCE", d.min_risk_confidence),
        red_flag_threshold=_float("SL_RED_FLAG_THRESHOLD", d.red_flag_threshold),
        typesafe_api_key=os.environ.get("TYPESAFE_API_KEY") or None,
        jev_base_url=os.environ.get("SL_JEV_BASE_URL", d.jev_base_url),
        jev_model=os.environ.get("SL_JEV_MODEL", d.jev_model),
        jev_timeout_s=_float("SL_JEV_TIMEOUT_S", d.jev_timeout_s),
        jev_max_retries=_int("SL_JEV_MAX_RETRIES", d.jev_max_retries),
        db_url=os.environ.get("SL_DB_URL", d.db_url),
    )
