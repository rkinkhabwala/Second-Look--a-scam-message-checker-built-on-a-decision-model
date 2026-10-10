"""Runtime settings, read from environment variables with typed defaults.

Thresholds live here (not in verdict.py) so they can be tuned from the eval's
coverage/accuracy curve without touching logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

API_ROOT = Path(__file__).resolve().parent.parent

ENGINE_THRESHOLDS: dict[str, dict[str, float]] = {}


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # Verdict thresholds (spec §7). Tuned 2026-10-10 on the Jev SMS calibration split and
    # checked on test: eval/results/2026-10-10-jev-sms-calibration-sweep.json.
    scam_threshold: float = 0.85
    safe_threshold: float = 0.25
    min_risk_confidence: float = 0.0  # 0 disables the risk_level confidence gate
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

    # Local engines (spec §3). Revisions pinned so results are reproducible.
    # Qwen3-1.7B, not 4B: on a 16 GB M1 Pro the 8 GB 4B model swapped (9–51 s per check);
    # 1.7B in fp16 runs ~2 s. Set SL_LLM_MODEL/SL_LLM_REVISION to try a larger model.
    llm_model: str = "Qwen/Qwen3-1.7B"
    llm_revision: str = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    nli_model: str = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c"
    nli_revision: str = "b2730f16019076bb0009481121efbe4705e0e378"
    device: str | None = None  # None = auto (mps > cuda > cpu)

    # Per-engine verdict thresholds, set from each engine's sweep (eval/results/*-sweep.json).
    # Engines not listed use the defaults above.
    engine_thresholds: dict[str, dict[str, float]] = field(default_factory=lambda: dict(ENGINE_THRESHOLDS))

    # Storage
    db_url: str = field(default_factory=lambda: f"sqlite:///{API_ROOT / 'second_look.db'}")


    def for_engine(self, engine: str) -> Settings:
        """These settings with the engine's own tuned thresholds applied, if it has any."""
        overrides = self.engine_thresholds.get(engine)
        return replace(self, **overrides) if overrides else self


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
        llm_model=os.environ.get("SL_LLM_MODEL", d.llm_model),
        llm_revision=os.environ.get("SL_LLM_REVISION", d.llm_revision),
        nli_model=os.environ.get("SL_NLI_MODEL", d.nli_model),
        nli_revision=os.environ.get("SL_NLI_REVISION", d.nli_revision),
        device=os.environ.get("SL_DEVICE") or None,
        db_url=os.environ.get("SL_DB_URL", d.db_url),
    )
