"""The hand-labelled recent-scam set (spec §9): format, PII masking and validation.

File: second_look/eval/data/handlabeled/handlabeled.jsonl (gitignored). One JSON object per line:

    {"text": "...", "label": "scam", "scam_type": "toll_or_fine", "channel": "sms",
     "received": "2026-09", "source": "own phone"}

- label: "scam" or "safe". scam_type: required for scams (an option from questions.py);
  leave it out (or use "not_a_scam") for safe messages.
- channel: "sms", "email" or "dm". received: year-month. source: where it came from.
- Include real legitimate messages too (bank alerts, delivery updates, 2FA codes),
  so the set can measure false alarms. Aim: 150-300 scams and ~100 safe.

    python -m second_look.eval.handlabeled validate          # check the file, flag leftover PII
    python -m second_look.eval.handlabeled mask              # mask phones/emails in place (keeps a .bak)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from second_look.eval.download import DATA_DIR
from second_look.questions import QUESTIONS
from second_look.storage import mask_pii

HANDLABELED_FILE = "handlabeled/handlabeled.jsonl"
SCAM_TYPES = set(QUESTIONS["scam_type"].criteria)
CHANNELS = {"sms", "email", "dm"}
REQUIRED = ("text", "label", "channel", "received", "source")

_LONG_DIGITS = re.compile(r"(?<!\d)\d{6,}(?!\d)")
_HANDLE = re.compile(r"(?<![\w.])@[A-Za-z0-9_]{3,}")
_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_GREETING_NAME = re.compile(r"\b(?:Hi|Hello|Dear|Hey)\s+([A-Z][a-z]+)\b")
_PLACEHOLDER_NAMES = {"Customer", "Sir", "Madam", "User", "Member", "Friend", "There", "Valued", "Mom", "Dad"}


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_record(r: dict) -> tuple[list[str], list[str]]:
    """(errors, warnings) for one record. Errors block use; warnings need a human look."""
    errors, warnings = [], []
    for f in REQUIRED:
        if not str(r.get(f, "")).strip():
            errors.append(f"missing '{f}'")
    if r.get("label") not in ("scam", "safe"):
        errors.append(f"label must be 'scam' or 'safe', got {r.get('label')!r}")
    st = r.get("scam_type")
    if r.get("label") == "scam":
        if st not in SCAM_TYPES - {"not_a_scam"}:
            errors.append(f"scams need a scam_type from questions.py, got {st!r}")
    elif r.get("label") == "safe" and st not in (None, "", "not_a_scam"):
        errors.append(f"safe messages can't have scam_type {st!r}")
    if r.get("channel") not in CHANNELS:
        errors.append(f"channel must be one of {sorted(CHANNELS)}")
    if r.get("received") and not _MONTH.match(str(r["received"])):
        errors.append("received must be YYYY-MM")
    text = str(r.get("text", ""))
    if mask_pii(text) != text:
        errors.append("contains a phone number or email address (run: ... handlabeled mask)")
    if _LONG_DIGITS.search(text):
        warnings.append("long digit run: check it isn't an account, card or tracking number")
    if _HANDLE.search(text):
        warnings.append("@handle: check it isn't a real person's account")
    names = {m for m in _GREETING_NAME.findall(text) if m not in _PLACEHOLDER_NAMES}
    if names:
        warnings.append(f"possible personal name after a greeting: {', '.join(sorted(names))}")
    return errors, warnings


def validate(path: Path) -> dict:
    records = load_records(path)
    report = {"n": len(records), "errors": [], "warnings": [], "duplicates": 0}
    seen: set[str] = set()
    for i, r in enumerate(records, 1):
        errors, warnings = check_record(r)
        report["errors"] += [f"line {i}: {e}" for e in errors]
        report["warnings"] += [f"line {i}: {w}" for w in warnings]
        key = str(r.get("text", "")).strip()
        if key in seen:
            report["duplicates"] += 1
            report["warnings"].append(f"line {i}: duplicate of an earlier message")
        seen.add(key)
    labels = Counter(r.get("label") for r in records)
    report["labels"] = dict(labels)
    report["scam_types"] = dict(Counter(r.get("scam_type") for r in records if r.get("label") == "scam"))
    if labels.get("scam", 0) < 150:
        report["warnings"].append(f"only {labels.get('scam', 0)} scams; aim for 150-300")
    if labels.get("safe", 0) < 100:
        report["warnings"].append(f"only {labels.get('safe', 0)} safe messages; aim for ~100")
    return report


def mask_file(path: Path) -> int:
    """Mask phones and emails in every record's text, in place. Returns how many changed."""
    records = load_records(path)
    changed = 0
    for r in records:
        masked = mask_pii(str(r.get("text", "")))
        if masked != r.get("text"):
            r["text"] = masked
            changed += 1
    if changed:
        shutil.copy(path, path.with_suffix(".jsonl.bak"))
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m second_look.eval.handlabeled", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["validate", "mask"])
    parser.add_argument("path", nargs="?", type=Path, default=DATA_DIR / HANDLABELED_FILE)
    args = parser.parse_args(argv)
    if not args.path.exists():
        print(f"{args.path} not found. Create it in the format shown in --help.")
        return 1
    if args.action == "mask":
        print(f"Masked {mask_file(args.path)} records (backup: {args.path.with_suffix('.jsonl.bak').name}).")
        return 0
    report = validate(args.path)
    print(f"{report['n']} records | labels {report['labels']} | scam types {report['scam_types']}")
    for e in report["errors"]:
        print(f"ERROR   {e}")
    for w in report["warnings"]:
        print(f"WARNING {w}")
    print("OK" if not report["errors"] else f"{len(report['errors'])} errors")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
