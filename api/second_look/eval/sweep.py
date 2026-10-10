"""Tune the verdict thresholds from saved predictions (spec §7: "tuned from the coverage/accuracy curve").

Tune on the calibration split and check on the test split, so the reported test
numbers aren't inflated by choosing thresholds on the data they're scored on:

    python -m second_look.eval.sweep \\
        --tune  second_look/eval/results/<date>-jev-sms-calibration-predictions.jsonl \\
        --check second_look/eval/results/<date>-jev-sms-predictions.jsonl

Makes no model calls: verdicts depend only on P(scam) and risk_level confidence,
both saved per message, and are recomputed with verdict.classify itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from second_look.config import Settings, get_settings
from second_look.eval.run import RESULTS_DIR
from second_look.verdict import VerdictLabel, classify

SAFE_GRID = [round(x, 2) for x in np.arange(0.05, 0.50, 0.05)]
SCAM_GRID = [round(x, 2) for x in np.arange(0.55, 1.00, 0.05)]
CONF_GRID = [round(x, 2) for x in np.arange(0.0, 0.65, 0.05)]


def load_predictions(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("error") is None]


def evaluate_policy(records: list[dict], settings: Settings) -> dict:
    """Coverage and errors of the verdict policy under the given thresholds."""
    n = len(records)
    counts = {"answered": 0, "correct": 0, "scams_called_safe": 0, "safe_called_scam": 0}
    for r in records:
        label = classify(r["p"], r.get("risk_confidence"), settings)
        if label is VerdictLabel.UNSURE:
            continue
        counts["answered"] += 1
        said_scam = label is VerdictLabel.LIKELY_SCAM
        if said_scam == bool(r["label"]):
            counts["correct"] += 1
        elif said_scam:
            counts["safe_called_scam"] += 1
        else:
            counts["scams_called_safe"] += 1
    answered = counts["answered"]
    return {
        "safe_threshold": settings.safe_threshold,
        "scam_threshold": settings.scam_threshold,
        "min_risk_confidence": settings.min_risk_confidence,
        "coverage": answered / n if n else 0.0,
        "accuracy_when_answered": counts["correct"] / answered if answered else None,
        "unsure_rate": 1 - answered / n if n else 0.0,
        **counts,
        "n": n,
    }


def sweep(records: list[dict], base: Settings) -> list[dict]:
    rows = []
    for safe in SAFE_GRID:
        for scam in SCAM_GRID:
            for conf in CONF_GRID:
                s = replace(base, safe_threshold=safe, scam_threshold=scam, min_risk_confidence=conf)
                rows.append(evaluate_policy(records, s))
    return rows


def recommend(rows: list[dict], min_accuracy: float) -> dict | None:
    """Highest coverage meeting the accuracy floor; ties go to fewer missed scams, then higher accuracy."""
    ok = [r for r in rows if r["accuracy_when_answered"] is not None and r["accuracy_when_answered"] >= min_accuracy]
    if not ok:
        return None
    return max(ok, key=lambda r: (round(r["coverage"], 6), -r["scams_called_safe"], r["accuracy_when_answered"]))


def as_settings(row: dict, base: Settings) -> Settings:
    return replace(base, safe_threshold=row["safe_threshold"], scam_threshold=row["scam_threshold"],
                   min_risk_confidence=row["min_risk_confidence"])


def _fmt(row: dict) -> str:
    acc = row["accuracy_when_answered"]
    return (f"safe≤{row['safe_threshold']:.2f} scam≥{row['scam_threshold']:.2f} conf≥{row['min_risk_confidence']:.2f}"
            f" | coverage {row['coverage']:.1%} | accuracy {'n/a' if acc is None else f'{acc:.2%}'}"
            f" | scams called safe {row['scams_called_safe']} | safe called scam {row['safe_called_scam']}")


def main(argv: list[str] | None = None, settings: Settings | None = None) -> dict:
    parser = argparse.ArgumentParser(prog="python -m second_look.eval.sweep", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tune", type=Path, required=True, help="predictions to choose thresholds on")
    parser.add_argument("--check", type=Path, help="held-out predictions to report the choice on")
    parser.add_argument("--min-accuracy", type=float, default=0.99)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    base = settings or get_settings()

    tune = load_predictions(args.tune)
    rows = sweep(tune, base)
    best = recommend(rows, args.min_accuracy)
    current_tune = evaluate_policy(tune, base)
    result = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tune_file": args.tune.name,
        "check_file": args.check.name if args.check else None,
        "min_accuracy": args.min_accuracy,
        "objective": "max coverage s.t. accuracy_when_answered >= min_accuracy; ties -> fewer scams called safe",
        "grid": {"safe": SAFE_GRID, "scam": SCAM_GRID, "min_risk_confidence": CONF_GRID},
        "current": {"tune": current_tune},
        "recommended": {"tune": best},
        "note": "Suggestion only: config is not changed automatically.",
        "rows": rows,
    }
    print(f"Tuned on {args.tune.name} (n={len(tune)}), accuracy floor {args.min_accuracy:.1%}")
    print(f"  current      {_fmt(current_tune)}")
    print(f"  recommended  {_fmt(best) if best else 'none meets the floor'}")
    if args.check:
        check = load_predictions(args.check)
        result["current"]["check"] = evaluate_policy(check, base)
        result["recommended"]["check"] = evaluate_policy(check, as_settings(best, base)) if best else None
        print(f"Checked on {args.check.name} (n={len(check)})")
        print(f"  current      {_fmt(result['current']['check'])}")
        if best:
            print(f"  recommended  {_fmt(result['recommended']['check'])}")

    out = args.out or RESULTS_DIR / args.tune.name.replace("-predictions.jsonl", "-sweep.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out}")
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
