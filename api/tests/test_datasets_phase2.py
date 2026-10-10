import json

import pytest

from second_look.eval.datasets import (
    Example,
    calibration_subset,
    load_handlabeled,
    load_phishing,
    load_split,
    stratified_sample,
)
from second_look.eval.download import PHISHING_FILE
from second_look.eval.handlabeled import HANDLABELED_FILE, check_record, main, mask_file, validate


def write_phishing(data_dir, n_safe=300, n_phish=100):
    path = data_dir / PHISHING_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [("", f"hello team, meeting notes {i}", "Safe Email") for i in range(n_safe)]
    rows += [("", f"verify your account now {i}", "Phishing Email") for i in range(n_phish)]
    rows += [("", "", "Safe Email"), ("", "empty", "Phishing Email"), ("", "x" * 60_000, "Safe Email"),
             ("", "hello team, meeting notes 0", "Safe Email")]  # blank, literal empty, too long, duplicate
    lines = ["Unnamed: 0,Email Text,Email Type"] + [f'{i},"{t}",{k}' for i, (_, t, k) in enumerate(rows)]
    path.write_text("\n".join(lines) + "\n")


def test_phishing_cleaning(tmp_path):
    write_phishing(tmp_path)
    exs = load_phishing(tmp_path, sample=None)
    assert len(exs) == 400
    assert sum(e.label for e in exs) == 100
    assert all(e.channel == "email" and e.text not in ("", "empty") and len(e.text) <= 50_000 for e in exs)


def test_phishing_sample_is_fixed_and_stratified(tmp_path):
    write_phishing(tmp_path)
    a, b = load_phishing(tmp_path, sample=200), load_phishing(tmp_path, sample=200)
    assert [e.id for e in a] == [e.id for e in b]
    assert len(a) == 200 and sum(e.label for e in a) == 50


def test_phishing_split_files_are_separate(tmp_path):
    write_phishing(tmp_path)
    test = load_split("phishing", "test", data_dir=tmp_path, splits_dir=tmp_path / "splits")
    assert (tmp_path / "splits" / "phishing.json").exists()
    assert len(test) == 160  # 40% of the 400 cleaned (sample of 2,000 > available)


def test_stratified_sample_order_independent():
    exs = [Example(id=f"x{i:03d}", text=str(i), label=int(i % 4 == 0)) for i in range(400)]
    a = stratified_sample(exs, 100)
    b = stratified_sample(list(reversed(exs)), 100)
    assert sorted(e.id for e in a) == sorted(e.id for e in b)
    assert sum(e.label for e in a) == 25


def test_calibration_subset():
    exs = [Example(id=f"x{i}", text=str(i), label=i % 2) for i in range(3000)]
    sub = calibration_subset(exs, 1000)
    assert len(sub) == 1000 and sum(e.label for e in sub) == 500
    assert calibration_subset(exs[:50], 1000) == exs[:50]


# ----------------------------------------------------------- hand-labelled


GOOD_SCAM = {"text": "E-ZPass: unpaid toll of $4.15, pay at ezpass-pay.top", "label": "scam",
             "scam_type": "toll_or_fine", "channel": "sms", "received": "2026-09", "source": "own phone"}
GOOD_SAFE = {"text": "Your verification code is 4821.", "label": "safe", "channel": "sms",
             "received": "2026-09", "source": "own phone"}


def test_good_records_pass():
    assert check_record(GOOD_SCAM) == ([], [])
    assert check_record(GOOD_SAFE) == ([], [])


@pytest.mark.parametrize(
    "change, message",
    [
        ({"label": "maybe"}, "label must be"),
        ({"scam_type": None}, "scams need a scam_type"),
        ({"scam_type": "not_a_scam"}, "scams need a scam_type"),
        ({"scam_type": "parking"}, "scams need a scam_type"),
        ({"channel": "fax"}, "channel must be"),
        ({"received": "Sept 2026"}, "YYYY-MM"),
        ({"source": ""}, "missing 'source'"),
        ({"text": "Call me at (555) 123-4567"}, "phone number or email"),
        ({"text": "write to jane.doe@gmail.com"}, "phone number or email"),
    ],
)
def test_record_errors(change, message):
    errors, _ = check_record({**GOOD_SCAM, **change})
    assert any(message in e for e in errors), errors


def test_safe_with_scam_type_is_error():
    errors, _ = check_record({**GOOD_SAFE, "scam_type": "romance"})
    assert errors


@pytest.mark.parametrize(
    "text, warning",
    [
        ("Your account 12345678 is locked", "long digit run"),
        ("DM @realperson_99 for details", "@handle"),
        ("Hi Priya, your parcel is waiting", "personal name"),
    ],
)
def test_pii_warnings(text, warning):
    _, warnings = check_record({**GOOD_SCAM, "text": text})
    assert any(warning in w for w in warnings), warnings


def test_placeholder_greeting_is_not_a_name():
    _, warnings = check_record({**GOOD_SCAM, "text": "Dear Customer, your account is locked"})
    assert warnings == []


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_validate_counts_and_duplicates(tmp_path):
    path = tmp_path / "h.jsonl"
    write_jsonl(path, [GOOD_SCAM, GOOD_SCAM, GOOD_SAFE])
    report = validate(path)
    assert report["n"] == 3 and report["duplicates"] == 1
    assert report["labels"] == {"scam": 2, "safe": 1}
    assert any("aim for 150-300" in w for w in report["warnings"])
    assert report["errors"] == []


def test_mask_file_in_place_with_backup(tmp_path):
    path = tmp_path / "h.jsonl"
    write_jsonl(path, [{**GOOD_SCAM, "text": "Call (555) 123-4567 or jo@x.com now"}, GOOD_SAFE])
    assert mask_file(path) == 1
    assert json.loads(path.read_text().splitlines()[0])["text"] == "Call [phone] or [email] now"
    assert path.with_suffix(".jsonl.bak").exists()
    assert validate(path)["errors"] == []


def test_cli_exit_codes(tmp_path, capsys):
    path = tmp_path / "h.jsonl"
    write_jsonl(path, [GOOD_SCAM])
    assert main(["validate", str(path)]) == 0
    write_jsonl(path, [{**GOOD_SCAM, "label": "x"}])
    assert main(["validate", str(path)]) == 1
    assert main(["validate", str(tmp_path / "missing.jsonl")]) == 1


def test_load_handlabeled(tmp_path):
    write_jsonl(tmp_path / HANDLABELED_FILE, [GOOD_SCAM, GOOD_SAFE, GOOD_SCAM])
    exs = load_handlabeled(tmp_path)
    assert len(exs) == 2
    scam = next(e for e in exs if e.label == 1)
    safe = next(e for e in exs if e.label == 0)
    assert scam.scam_type == "toll_or_fine" and safe.scam_type == "not_a_scam"
    assert scam.channel == "sms"
