"""Compare engines on each dataset from their latest full test-split results.

    python -m second_look.eval.compare

Writes eval/results/<date>-comparison.json and one reliability grid per dataset.
Refuses to compare results scored on different messages (split_hash mismatch).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from second_look.eval import charts
from second_look.eval.run import RESULTS_DIR

ENGINES = ("jev", "llm", "nli")
_RESULT = re.compile(r"^(\d{4}-\d{2}-\d{2})-(jev|llm|nli)-([a-z]+)\.json$")


def latest_results(results_dir: Path) -> dict[tuple[str, str], dict]:
    """(engine, dataset) -> the newest full test-split summary."""
    found: dict[tuple[str, str], tuple[str, dict]] = {}
    for path in results_dir.glob("*.json"):
        m = _RESULT.match(path.name)
        if not m:
            continue  # limit runs, calibration splits, sweeps, comparisons
        date, engine, dataset = m.groups()
        summary = json.loads(path.read_text())
        if summary.get("limit") or summary.get("split") != "test":
            continue
        if (engine, dataset) not in found or date >= found[(engine, dataset)][0]:
            found[(engine, dataset)] = (date, summary)
    return {k: v[1] for k, v in found.items()}


def row(summary: dict) -> dict:
    s = summary["is_scam"]
    raw = (summary.get("uncalibrated") or {}).get("is_scam") or {}
    policy = summary.get("verdict_policy") or {}
    return {
        "engine": summary["engine"],
        "model": summary["model"],
        "n": summary["n_scored"],
        "split_hash": summary.get("split_hash"),
        "calibrated": "calibration" in summary,
        "accuracy_at_0.5": s["accuracy_at_0.5"],
        "roc_auc": s["roc_auc"],
        "ece": s["ece_10_bins"],
        "ece_uncalibrated": raw.get("ece_10_bins"),
        "brier": s["brier"],
        "accuracy_at_80_coverage": s["accuracy_at_80_coverage"],
        "accuracy_at_90_coverage": s["accuracy_at_90_coverage"],
        "policy_coverage": policy.get("coverage"),
        "policy_accuracy": policy.get("accuracy_when_answered"),
        "scam_type_macro_f1": summary["scam_type"].get("macro_f1"),
        "latency_p50_ms": summary["latency_ms"]["p50"],
        "latency_p95_ms": summary["latency_ms"]["p95"],
        "cost_per_1000_usd": summary["cost_per_1000_checks"]["usd"] or 0.0,
        "thresholds": summary["thresholds"],
    }


def compare(results_dir: Path = RESULTS_DIR, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or results_dir
    latest = latest_results(results_dir)
    datasets = sorted({d for _, d in latest})
    out = {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), "datasets": {}}
    date = out["created_at"][:10]
    for dataset in datasets:
        rows = [row(latest[(e, dataset)]) for e in ENGINES if (e, dataset) in latest]
        hashes = {r["split_hash"] for r in rows if r["split_hash"]}
        if len(hashes) > 1:
            raise SystemExit(f"{dataset}: engines were scored on different messages ({sorted(hashes)})")
        out["datasets"][dataset] = {"same_split": len(hashes) <= 1, "split_hash": next(iter(hashes), None),
                                    "engines": rows}
        panels = [(f"{e} ({latest[(e, dataset)]['model']})", latest[(e, dataset)]["is_scam"]["reliability_bins"],
                   latest[(e, dataset)]["is_scam"]["ece_10_bins"]) for e in ENGINES if (e, dataset) in latest]
        chart = out_dir / f"{date}-comparison-{dataset}-reliability.png"
        charts.reliability_grid(panels, f"Reliability by engine — {dataset} test split", chart)
        out["datasets"][dataset]["chart"] = chart.name
    path = out_dir / f"{date}-comparison.json"
    path.write_text(json.dumps(out, indent=2))
    out["path"] = str(path)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m second_look.eval.compare", description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    out = compare(args.results_dir)
    for dataset, d in out["datasets"].items():
        print(f"\n{dataset} (n={d['engines'][0]['n']}, same split: {d['same_split']})")
        print(f"  {'engine':6} {'acc':>6} {'AUC':>6} {'ECE':>6} {'ECEraw':>6} {'acc@90':>7} {'p50 ms':>7} {'$/1k':>7}")
        for r in d["engines"]:
            raw = "" if r["ece_uncalibrated"] is None else f"{r['ece_uncalibrated']:.3f}"
            print(f"  {r['engine']:6} {r['accuracy_at_0.5']:6.3f} {r['roc_auc']:6.3f} {r['ece']:6.3f} {raw:>6} "
                  f"{r['accuracy_at_90_coverage']:7.3f} {r['latency_p50_ms']:7.0f} {r['cost_per_1000_usd']:7.4f}")
    print(f"\nWrote {out['path']}")


if __name__ == "__main__":
    main(sys.argv[1:])
