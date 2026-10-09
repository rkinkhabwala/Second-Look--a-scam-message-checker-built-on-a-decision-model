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

from second_look.eval.download import DATA_DIR, SMS_FILE

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


LOADERS = {"sms": load_sms}

DATASET_NOTES = {
    "sms": (
        "UCI SMS Spam Collection, exact duplicates removed. Positive label = 'spam', which includes "
        "legitimate-but-unwanted marketing, so a model that correctly calls marketing 'not a scam' "
        "is scored as wrong on those messages. No scam_type labels, so scam_type macro-F1 is not computable."
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
