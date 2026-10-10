import json
from dataclasses import replace

import pytest

from second_look.config import Settings
from second_look.eval.compare import compare
from second_look.eval.run import main
from second_look.engines.llm_scoring import LLMScoringEngine
from tests.test_eval import cli_args, write_sms
from tests.test_datasets_phase2 import GOOD_SAFE, GOOD_SCAM, write_jsonl


class OverconfidentScorer:
    """Right ~80% of the time but always ~certain: temperature scaling should cool it down."""

    model_id = "fake/llm@00000000"

    def letter_logits(self, system, context, suffixes, n_options):
        scammy = "WIN" in context or "toll" in context
        digit = sum(map(ord, context)) % 5  # deterministic "noise": flip 1 in 5
        right = digit != 0
        says_scam = scammy if right else not scammy
        out = []
        for k in n_options:
            row = [0.0] * k
            row[0 if says_scam else min(1, k - 1)] = 9.0
            out.append(row)
        return out


def local_factory(name, settings):
    return LLMScoringEngine(OverconfidentScorer())


def test_calibrate_reports_raw_and_calibrated_and_lowers_ece(tmp_path):
    write_sms(tmp_path / "data", n_ham=300, n_spam=120)
    args = cli_args(tmp_path, "--engine", "llm", "--calibrate", "--calibration-dir", str(tmp_path / "cal"))
    summary = main(args, engine_factory=local_factory, settings=Settings())
    cal = summary["calibration"]
    assert cal["temperatures"]["is_scam"] > 1.5  # overconfident -> T > 1
    assert cal["fit_examples"] == 252  # whole calibration split (< 1,000)
    assert summary["is_scam"]["ece_10_bins"] < summary["uncalibrated"]["is_scam"]["ece_10_bins"]
    assert {p.name for p in (tmp_path / "cal").iterdir()} == {"llm-sms.json", "llm.json"}
    assert summary["charts"]["reliability_uncalibrated"].endswith("-reliability-uncalibrated.png")
    preds = (tmp_path / "results").glob("*-llm-sms-predictions.jsonl")
    assert '"answers"' not in next(preds).read_text()


def test_calibrate_only_on_test_split(tmp_path):
    write_sms(tmp_path / "data")
    with pytest.raises(SystemExit):
        main(cli_args(tmp_path, "--calibrate", "--split", "calibration"), engine_factory=local_factory)


def test_scam_type_macro_f1_with_handlabeled(tmp_path):
    from second_look.eval.handlabeled import HANDLABELED_FILE

    recs = [{**GOOD_SCAM, "text": f"toll due {i}"} for i in range(30)] + \
           [{**GOOD_SAFE, "text": f"see you at {i}"} for i in range(30)]
    write_jsonl(tmp_path / "data" / HANDLABELED_FILE, recs)
    args = ["--engine", "llm", "--dataset", "handlabeled", "--data-dir", str(tmp_path / "data"),
            "--splits-dir", str(tmp_path / "splits"), "--cache-dir", str(tmp_path / "cache"),
            "--out-dir", str(tmp_path / "results")]
    summary = main(args, engine_factory=local_factory, settings=Settings())
    st = summary["scam_type"]
    assert st["n_labelled"] == 24 and st["macro_f1"] is not None and 0 <= st["macro_f1"] <= 1


def test_engine_thresholds_apply_per_engine():
    s = Settings(engine_thresholds={"nli": {"scam_threshold": 0.95}})
    assert s.for_engine("nli").scam_threshold == 0.95
    assert s.for_engine("jev") is s


def test_compare_checks_same_split_and_writes_outputs(tmp_path):
    write_sms(tmp_path / "data", n_ham=200, n_spam=60)
    for engine in ("llm", "jev"):
        main(cli_args(tmp_path, "--engine", engine), engine_factory=local_factory, settings=Settings())
    # local_factory always returns the llm engine; relabel one file as jev to simulate two engines
    res = tmp_path / "results"
    llm_json = next(p for p in res.glob("*-llm-sms.json"))
    jev = json.loads(llm_json.read_text()); jev["engine"] = "jev"
    (res / llm_json.name.replace("-llm-", "-jev-")).write_text(json.dumps(jev))
    out = compare(res)
    d = out["datasets"]["sms"]
    assert d["same_split"] and [r["engine"] for r in d["engines"]] == ["jev", "llm"]
    assert (res / d["chart"]).exists()

    jev["split_hash"] = "different"
    (res / llm_json.name.replace("-llm-", "-jev-")).write_text(json.dumps(jev))
    with pytest.raises(SystemExit, match="different messages"):
        compare(res)
