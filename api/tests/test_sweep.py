import json

import pytest

from second_look.config import Settings
from second_look.eval.run import write_outputs
from second_look.eval.sweep import evaluate_policy, main, recommend, sweep
from tests.test_verdict import SPEC_S7

# p, risk_confidence, label
RECORDS = [
    {"p": 0.95, "risk_confidence": 0.9, "label": 1},
    {"p": 0.90, "risk_confidence": 0.3, "label": 1},  # gated out by conf 0.4, right if answered
    {"p": 0.60, "risk_confidence": 0.9, "label": 1},  # in the default band
    {"p": 0.05, "risk_confidence": 0.9, "label": 0},
    {"p": 0.10, "risk_confidence": 0.2, "label": 0},  # gated out, right if answered
    {"p": 0.30, "risk_confidence": 0.9, "label": 1},  # a scam that a wide "safe" band would miss
]


def test_evaluate_policy_matches_verdict_defaults():
    out = evaluate_policy(RECORDS, SPEC_S7)
    assert out["answered"] == 2 and out["correct"] == 2
    assert out["coverage"] == pytest.approx(2 / 6)
    assert out["unsure_rate"] == pytest.approx(4 / 6)


def test_evaluate_policy_counts_error_kinds():
    s = Settings(safe_threshold=0.35, scam_threshold=0.55, min_risk_confidence=0.0)
    out = evaluate_policy(RECORDS + [{"p": 0.99, "risk_confidence": 0.9, "label": 0}], s)
    assert out["scams_called_safe"] == 1  # p=0.30 scam
    assert out["safe_called_scam"] == 1  # p=0.99 ham
    assert out["answered"] == 7


def test_recommend_maximizes_coverage_under_accuracy_floor():
    best = recommend(sweep(RECORDS, Settings()), min_accuracy=1.0)
    assert best["accuracy_when_answered"] == 1.0
    assert best["min_risk_confidence"] <= 0.2  # dropping the gate is free here
    assert best["safe_threshold"] < 0.30 and best["scam_threshold"] <= 0.60
    assert best["coverage"] == pytest.approx(5 / 6)


def test_recommend_none_when_floor_unreachable():
    rows = sweep([{"p": 0.99, "risk_confidence": 0.9, "label": 0}], Settings())
    assert recommend(rows, 0.5) is None


def test_missing_risk_confidence_is_not_gated():
    out = evaluate_policy([{"p": 0.9, "risk_confidence": None, "label": 1}], Settings())
    assert out["answered"] == 1


def write_preds(path, records):
    path.write_text("".join(json.dumps({"error": None, **r}) + "\n" for r in records)
                    + json.dumps({"error": "HTTP 500", "label": 1}) + "\n")


def test_cli_tunes_and_checks_and_writes_json(tmp_path, capsys):
    tune, check = tmp_path / "x-calibration-predictions.jsonl", tmp_path / "x-predictions.jsonl"
    write_preds(tune, RECORDS)
    write_preds(check, RECORDS[:3])
    out = main(["--tune", str(tune), "--check", str(check), "--min-accuracy", "1.0",
                "--out", str(tmp_path / "sweep.json")], settings=SPEC_S7)
    assert out["recommended"]["tune"]["n"] == 6  # errored rows dropped
    assert out["recommended"]["check"]["coverage"] == pytest.approx(1.0)
    assert out["current"]["check"]["coverage"] == pytest.approx(1 / 3)
    assert json.loads((tmp_path / "sweep.json").read_text())["note"].startswith("Suggestion only")
    assert "recommended" in capsys.readouterr().out


def test_calibration_results_do_not_overwrite_test_results(tmp_path):
    base = {"created_at": "2026-10-10T00:00:00", "engine": "jev", "dataset": "sms", "limit": None,
            "model": "m", "n_scored": 0, "is_scam": {"reliability_bins": []}}
    a = write_outputs({**base, "split": "test"}, [], tmp_path)
    b = write_outputs({**base, "split": "calibration"}, [], tmp_path)
    assert a["json"].name == "2026-10-10-jev-sms.json"
    assert b["json"].name == "2026-10-10-jev-sms-calibration.json"
