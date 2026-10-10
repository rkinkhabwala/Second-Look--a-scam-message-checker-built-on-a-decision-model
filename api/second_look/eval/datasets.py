"""Eval datasets and the fixed, seeded split (spec §9: 60% calibration-fit / 40% test).

The split is written to eval/splits/<dataset>.json and committed, so every
engine is evaluated on exactly the same test messages.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

import pandas as pd

from second_look.eval.download import DATA_DIR, PHISHING_FILE, SMS_FILE

EVAL_DIR = Path(__file__).resolve().parent
SPLITS_DIR = EVAL_DIR / "splits"
SEED = 42
TEST_SIZE = 0.4


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: int  # 1 = scam/spam, 0 = legitimate
    channel: str = "sms"
    scam_type: str | None = None  # only where a dataset labels it (hand-labelled set)


def text_id(dataset: str, text: str) -> str:
    """Stable ID from content, so splits survive re-downloads and row reordering."""
    return f"{dataset}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"


def load_sms(data_dir: Path = DATA_DIR) -> list[Example]:
    """UCI SMS Spam Collection. 'spam' (which includes marketing spam) counts as positive.

    Exact-duplicate messages are dropped (keeping the first) so no message can
    appear in both the calibration and test splits.
    """
    path = data_dir / SMS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m second_look.eval.download sms")
    examples: dict[str, Example] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        label, text = line.split("\t", 1)
        text = text.strip()
        ex_id = text_id("sms", text)
        if ex_id not in examples:
            examples[ex_id] = Example(id=ex_id, text=text, label=int(label == "spam"))
    return list(examples.values())


PHISHING_SAMPLE = 2_000
PHISHING_MAX_CHARS = 50_000


def load_phishing(data_dir: Path = DATA_DIR, sample: int | None = PHISHING_SAMPLE) -> list[Example]:
    """zefang-liu/phishing-email-dataset, cleaned, then a fixed stratified sample (PLAN D5).

    Drops null/blank/"empty" rows, exact duplicates and rows over 50k characters.
    'Phishing Email' (which includes generic spam) counts as positive.
    """
    path = data_dir / PHISHING_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m second_look.eval.download phishing")
    df = pd.read_csv(path, usecols=["Email Text", "Email Type"])
    text = df["Email Text"].fillna("").astype(str).str.strip()
    keep = (text != "") & (text.str.lower() != "empty") & (text.str.len() <= PHISHING_MAX_CHARS)
    examples: dict[str, Example] = {}
    for t, kind in zip(text[keep], df["Email Type"][keep]):
        ex_id = text_id("phishing", t)
        if ex_id not in examples:
            examples[ex_id] = Example(id=ex_id, text=t, label=int(kind == "Phishing Email"), channel="email")
    out = list(examples.values())
    if sample and sample < len(out):
        out = stratified_sample(out, sample)
    return out


def stratified_sample(examples: list[Example], n: int, seed: int = SEED) -> list[Example]:
    """A fixed, label-stratified subset (order-independent: IDs are sorted first)."""
    ordered = sorted(examples, key=lambda e: e.id)
    keep, _ = train_test_split(ordered, train_size=n, random_state=seed, stratify=[e.label for e in ordered])
    return keep


def calibration_subset(examples: list[Example], n: int = 1_000) -> list[Example]:
    """The fixed subset of a calibration split that local engines fit temperatures on (PLAN D3)."""
    return examples if len(examples) <= n else stratified_sample(examples, n)


def load_handlabeled(data_dir: Path = DATA_DIR) -> list[Example]:
    from second_look.eval.handlabeled import HANDLABELED_FILE, load_records

    path = data_dir / HANDLABELED_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See: python -m second_look.eval.handlabeled --help")
    out: dict[str, Example] = {}
    for r in load_records(path):
        ex_id = text_id("hand", r["text"])
        out.setdefault(ex_id, Example(
            id=ex_id, text=r["text"], label=int(r["label"] == "scam"), channel=r["channel"],
            scam_type=r.get("scam_type") or ("not_a_scam" if r["label"] == "safe" else None),
        ))
    return list(out.values())


LOADERS = {"sms": load_sms, "phishing": load_phishing, "handlabeled": load_handlabeled}

DATASET_NOTES = {
    "sms": (
        "UCI SMS Spam Collection, exact duplicates removed. Positive label = 'spam', which includes "
        "legitimate-but-unwanted marketing, so a model that correctly calls marketing 'not a scam' "
        "is scored as wrong on those messages. No scam_type labels, so scam_type macro-F1 is not computable."
    ),
    "phishing": (
        "zefang-liu/phishing-email-dataset (Kaggle 'Phishing Email Detection'; Enron-era mail). Null/empty rows, "
        "exact duplicates and rows over 50k characters removed; fixed stratified sample of 2,000. Positive label "
        "'Phishing Email' includes generic spam. No scam_type labels."
    ),
    "handlabeled": (
        "Hand-labelled recent messages (scams and legitimate), personal data masked. Has scam_type labels."
    ),
}


def make_split(examples: list[Example], seed: int = SEED) -> dict[str, list[str]]:
    ids = sorted(e.id for e in examples)
    by_id = {e.id: e for e in examples}
    calib, test = train_test_split(
        ids, test_size=TEST_SIZE, random_state=seed, stratify=[by_id[i].label for i in ids]
    )
    return {"calibration": calib, "test": test}


def load_split(
    dataset: str,
    split: str = "test",
    *,
    data_dir: Path = DATA_DIR,
    splits_dir: Path = SPLITS_DIR,
) -> list[Example]:
    """Examples for one split, in the split file's (shuffled) order."""
    examples = LOADERS[dataset](data_dir)
    by_id = {e.id: e for e in examples}
    split_file = splits_dir / f"{dataset}.json"
    if split_file.exists():
        splits = json.loads(split_file.read_text())
    else:
        splits = make_split(examples)
        splits_dir.mkdir(parents=True, exist_ok=True)
        split_file.write_text(json.dumps({"seed": SEED, "test_size": TEST_SIZE, **splits}, indent=0))
    missing = [i for i in splits[split] if i not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} split IDs not found in the {dataset} data; re-download it")
    return [by_id[i] for i in splits[split]]
