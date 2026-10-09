import json

import numpy as np
import pytest

from second_look.config import Settings
from second_look.engines.base import EngineError
from second_look.engines.fake import FakeEngine
from second_look.eval import metrics
from second_look.eval.datasets import Example, load_split, make_split, text_id
from second_look.eval.download import SMS_FILE
from second_look.eval.run import evaluate, main, plan_run
from tests.test_verdict import answers

# ------------------------------------------------------------------ metrics


def test_accuracy_auc_brier_hand_computed():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    assert metrics.accuracy(y, p) == 0.5
    assert metrics.roc_auc(y, p) == 0.75
    assert metrics.brier(y, p) == pytest.approx((0.01 + 0.36 + 0.36 + 0.01) / 4)


def test_roc_auc_single_class_is_none():
    assert metrics.roc_auc(np.array([1, 1]), np.array([0.2, 0.9])) is None


def test_ece_zero_when_perfectly_calibrated():
    # bin 0.25-ish: 4 messages at p=0.25, one positive; bin 0.75: 4 at p=0.75, three positive
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    p = np.array([0.25] * 4 + [0.75] * 4)
    assert metrics.ece(y, p) == pytest.approx(0.0)


def test_ece_hand_computed():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.95, 0.95, 0.95, 0.95])  # one bin, mean 0.95, observed 0.5
    assert metrics.ece(y, p) == pytest.approx(0.45)


def test_reliability_bins_edges_and_counts():
    bins = metrics.reliability_bins(np.array([0, 1, 1]), np.array([0.0, 0.55, 1.0]))
    assert len(bins) == 10
    assert [b["count"] for b in bins] == [1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    assert bins[9]["mean_predicted"] == 1.0  # p = 1.0 lands in the last bin


def test_coverage_curve_answers_confident_first_and_is_monotone_on_sorted_data():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < p).astype(int)  # calibrated: confident = more accurate
    curve = metrics.coverage_curve(y, p)
    assert curve[-1]["coverage"] == 1.0
    assert curve[-1]["accuracy"] == pytest.approx(metrics.accuracy(y, p))
    assert curve[0]["accuracy"] >= curve[-1]["accuracy"]
    covs = [c["coverage"] for c in curve]
    assert covs == sorted(covs)
    bands = [c["band"][1] - c["band"][0] for c in curve]
    assert all(b2 <= b1 + 1e-12 for b1, b2 in zip(bands, bands[1:]))  # abstain band narrows as coverage grows


def test_accuracy_at_coverage_hand_computed():
    y = np.array([1, 0, 1, 0, 1])
    p = np.array([0.99, 0.01, 0.6, 0.55, 0.45])  # least confident two are both wrong
    assert metrics.accuracy_at_coverage(y, p, 0.6) == 1.0
    assert metrics.accuracy_at_coverage(y, p, 1.0) == pytest.approx(0.6)


def test_recommend_band():
    y = np.array([1, 0, 1, 0, 1])
    p = np.array([0.99, 0.01, 0.6, 0.55, 0.45])
    rec = metrics.recommend_band(y, p, 1.0)
    assert rec["coverage"] == pytest.approx(0.6) and rec["band"] == pytest.approx([0.4, 0.6])


def test_policy_metrics():
    out = metrics.policy_metrics(np.array([1, 0, 1, 0]), ["LIKELY_SCAM", "PROBABLY_SAFE", "UNSURE", "LIKELY_SCAM"])
    assert out["coverage"] == 0.75
    assert out["accuracy_when_answered"] == pytest.approx(2 / 3)
    assert out["counts"] == {"LIKELY_SCAM": 2, "UNSURE": 1, "PROBABLY_SAFE": 1}


def test_cost_per_1000():
    out = metrics.cost_per_1000([800, 1200], 0.042)
    assert out["usd"] == pytest.approx(1000 * 1000 * 0.042 / 1e6)


# -------------------------------------------------------------------- split


def write_sms(data_dir, n_ham=60, n_spam=15, dupes=True):
    path = data_dir / SMS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"ham\thello friend number {i}" for i in range(n_ham)]
    lines += [f"spam\tWIN $1000 now, text CLAIM to 8{i:04d}" for i in range(n_spam)]
    if dupes:
        lines += ["ham\thello friend number 0", "spam\tWIN $1000 now, text CLAIM to 80000"]
    path.write_text("\n".join(lines) + "\n")


def test_split_is_deterministic_stratified_and_disjoint(tmp_path):
    exs = [Example(id=f"x{i}", text=str(i), label=int(i % 5 == 0)) for i in range(200)]
    a, b = make_split(exs), make_split(list(reversed(exs)))
    assert a == b
    assert not set(a["test"]) & set(a["calibration"])
    assert len(a["test"]) == 80
    labels = {e.id: e.label for e in exs}
    assert sum(labels[i] for i in a["test"]) == 16  # 20% positives preserved


def test_load_split_dedupes_and_persists_split_file(tmp_path):
    write_sms(tmp_path / "data")
    test = load_split("sms", "test", data_dir=tmp_path / "data", splits_dir=tmp_path / "splits")
    calib = load_split("sms", "calibration", data_dir=tmp_path / "data", splits_dir=tmp_path / "splits")
    assert len(test) + len(calib) == 75  # 77 lines, 2 exact duplicates
    assert len({e.text for e in test + calib}) == 75
    saved = json.loads((tmp_path / "splits" / "sms.json").read_text())
    assert saved["seed"] == 42 and saved["test"] == [e.id for e in test]
    assert test[0].id == text_id("sms", test[0].text)


def test_committed_sms_split_has_expected_size():
    from second_look.eval.datasets import SPLITS_DIR
    saved = json.loads((SPLITS_DIR / "sms.json").read_text())
    assert len(saved["test"]) == 2064 and len(saved["calibration"]) == 3096
    assert not set(saved["test"]) & set(saved["calibration"])


# ------------------------------------------------------------- run / cache


def scripted(state, qs):
    p = 0.95 if "WIN" in state["message"] else 0.05
    return answers(p, scam_type="prize_or_lottery" if p > 0.5 else "not_a_scam")


async def test_cache_hit_avoids_second_engine_call(tmp_path):
    engine = FakeEngine(scripted)
    exs = [Example(id="a", text="WIN $5", label=1), Example(id="b", text="hi mum", label=0)]
    first = await evaluate(engine, exs, settings=Settings(), cache_dir=tmp_path, progress=lambda _: None)
    assert len(engine.calls) == 2 and not any(r["cached"] for r in first)
    second = await evaluate(engine, exs, settings=Settings(), cache_dir=tmp_path, progress=lambda _: None)
    assert len(engine.calls) == 2 and all(r["cached"] for r in second)
    assert [r["p"] for r in first] == [r["p"] for r in second]
    assert [r["latency_ms"] for r in first] == [r["latency_ms"] for r in second]
    third = await evaluate(engine, exs, settings=Settings(), cache_dir=tmp_path, use_cache=False,
                           progress=lambda _: None)
    assert len(engine.calls) == 4 and not any(r["cached"] for r in third)


async def test_engine_errors_are_recorded_not_cached(tmp_path):
    def flaky(state, qs):
        raise EngineError("HTTP 500", status=500)
    exs = [Example(id="a", text="x", label=1)]
    (rec,) = await evaluate(FakeEngine(flaky), exs, settings=Settings(), cache_dir=tmp_path, progress=lambda _: None)
    assert rec["error"] == "HTTP 500"
    assert list(tmp_path.iterdir()) == []


async def test_bad_key_stops_the_run(tmp_path):
    def unauthorized(state, qs):
        raise EngineError("HTTP 401", status=401)
    with pytest.raises(EngineError):
        await evaluate(FakeEngine(unauthorized), [Example(id="a", text="x", label=1)],
                       settings=Settings(), cache_dir=tmp_path, progress=lambda _: None)


def cli_args(tmp_path, *extra):
    return ["--engine", "fake", "--dataset", "sms", "--data-dir", str(tmp_path / "data"),
            "--splits-dir", str(tmp_path / "splits"), "--cache-dir", str(tmp_path / "cache"),
            "--out-dir", str(tmp_path / "results"), *extra]


def test_cli_end_to_end_writes_json_and_charts(tmp_path):
    write_sms(tmp_path / "data")
    engine = FakeEngine(scripted)
    summary = main(cli_args(tmp_path), engine_factory=lambda name, s: engine, settings=Settings())

    assert summary["n"] == 30 and summary["n_errors"] == 0
    s = summary["is_scam"]
    assert s["accuracy_at_0.5"] == 1.0 and s["roc_auc"] == 1.0
    for key in ("ece_10_bins", "brier", "accuracy_at_80_coverage", "accuracy_at_90_coverage"):
        assert s[key] is not None
    assert summary["scam_type"]["macro_f1"] is None
    assert summary["latency_ms"]["p50"] is not None and summary["latency_ms"]["p95"] is not None
    assert summary["cost_per_1000_checks"]["usd"] is None  # fake engine reports no tokens

    files = sorted(p.name for p in (tmp_path / "results").iterdir())
    assert len(files) == 4
    assert any(f.endswith("-fake-sms.json") for f in files)
    assert any(f.endswith("-reliability.png") for f in files)
    assert any(f.endswith("-coverage.png") for f in files)
    saved = json.loads(next((tmp_path / "results").glob("*-fake-sms.json")).read_text())
    assert saved["charts"]["reliability"].endswith(".png")
    preds = next((tmp_path / "results").glob("*-predictions.jsonl")).read_text()
    assert "hello friend" not in preds  # no message text in results


def test_cli_limit_and_dry_run(tmp_path):
    write_sms(tmp_path / "data")
    engine = FakeEngine(scripted)
    plan = main(cli_args(tmp_path, "--limit", "5", "--dry-run"), engine_factory=lambda n, s: engine,
                settings=Settings())
    assert plan["requests"] == 5 and engine.calls == []
    summary = main(cli_args(tmp_path, "--limit", "5"), engine_factory=lambda n, s: engine, settings=Settings())
    assert summary["n"] == 5 and len(engine.calls) == 5
    assert next((tmp_path / "results").glob("*-limit5.json"))
    plan = main(cli_args(tmp_path, "--limit", "5", "--dry-run"), engine_factory=lambda n, s: engine,
                settings=Settings())
    assert plan["requests"] == 0 and plan["cached"] == 5


def test_cli_without_key_exits_cleanly(tmp_path):
    write_sms(tmp_path / "data")
    with pytest.raises(SystemExit, match="not available"):
        main(cli_args(tmp_path, "--engine", "jev"), settings=Settings(typesafe_api_key=None))
