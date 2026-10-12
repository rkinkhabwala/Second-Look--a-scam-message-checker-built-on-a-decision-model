"""Run an engine over an eval split and write metrics JSON + charts.

    python -m second_look.eval.run --engine jev --dataset sms
    python -m second_look.eval.run --engine jev --dataset sms --limit 50
    python -m second_look.eval.run --engine jev --dataset sms --dry-run   # count requests & cost, no calls

Engine responses are cached on disk by request hash, so reruns don't pay twice.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from second_look.config import Settings, get_settings
from second_look.engines.base import Answer, DecisionEngine, EngineError, EngineUnavailable
from second_look.engines.registry import EngineProvider
from second_look.eval import charts, metrics
from second_look.calibration import Calibration, fit_calibration
from second_look.eval.datasets import DATASET_NOTES, EVAL_DIR, LOADERS, Example, calibration_subset, load_split
from second_look.eval.download import DATA_DIR
from second_look.preprocess import (
    BYTES_PER_TOKEN,
    MEASURED_JEV_BYTES_PER_TOKEN,
    build_state,
    estimate_tokens,
    state_token_budget,
)
from second_look.questions import QUESTIONS
from second_look.verdict import make_verdict

CACHE_DIR = EVAL_DIR / "cache"
RESULTS_DIR = EVAL_DIR / "results"
FATAL_STATUSES = {401, 403}  # bad key: stop instead of failing every message


# ----------------------------------------------------------------- requests


def _questions_json() -> dict:
    return {k: q.model_dump() for k, q in QUESTIONS.items()}


def make_request(engine: DecisionEngine, example: Example, settings: Settings) -> tuple[dict, dict, bool]:
    """(state, cache-key payload, truncated) for one example."""
    budget = state_token_budget(
        _questions_json(), settings.jev_max_state_tokens, settings.jev_max_request_tokens
    )
    state, meta = build_state(example.text, example.channel, max_state_tokens=budget)
    if hasattr(engine, "build_payload"):  # the engine's exact input, so caches key on model + prompt
        payload, engine_truncated = engine.build_payload(state, QUESTIONS)
        return state, payload, meta.truncated or engine_truncated
    return state, {"engine": engine.name, "state": state, "questions": _questions_json()}, meta.truncated


def split_hash(examples: list[Example]) -> str:
    """Fingerprint of exactly which messages were scored, so comparisons can prove they match."""
    return hashlib.sha256("\n".join(sorted(e.id for e in examples)).encode()).hexdigest()[:16]


def cache_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def plan_run(engine: DecisionEngine, examples: list[Example], cache_dir: Path, settings: Settings) -> dict:
    """How many uncached requests a run would make, and roughly what they'd cost."""
    uncached, tokens = 0, []
    for ex in examples:
        _, payload, _ = make_request(engine, ex, settings)
        if not (cache_dir / f"{cache_key(payload)}.json").exists():
            uncached += 1
            tokens.append(estimate_tokens(payload))
    conservative = sum(tokens)
    likely = int(conservative * BYTES_PER_TOKEN / MEASURED_JEV_BYTES_PER_TOKEN)
    price = settings.jev_price_per_mtok_usd if engine.name == "jev" else 0.0
    return {
        "examples": len(examples),
        "cached": len(examples) - uncached,
        "requests": uncached,
        "input_tokens_conservative": conservative,
        "input_tokens_likely": likely,
        "cost_usd_conservative": conservative * price / 1e6,
        "cost_usd_likely": likely * price / 1e6,
    }


# ------------------------------------------------------------------ running


async def _call(engine: DecisionEngine, state: dict) -> dict:
    start = time.perf_counter()
    if hasattr(engine, "run"):
        r = await engine.run(state, QUESTIONS)
        return {"answers": r.answers, "model": r.model, "usage": r.usage,
                "latency_ms": r.latency_ms, "attempts": r.attempts, "truncated": r.truncated}
    answers = await engine.decide(state, QUESTIONS)
    return {"answers": answers, "model": getattr(engine, "model", None), "usage": {},
            "latency_ms": (time.perf_counter() - start) * 1000, "attempts": 1, "truncated": False}


async def evaluate(
    engine: DecisionEngine,
    examples: list[Example],
    *,
    settings: Settings,
    cache_dir: Path = CACHE_DIR,
    use_cache: bool = True,
    concurrency: int = 8,
    progress: Callable[[str], None] = print,
) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(ex: Example) -> dict:
        nonlocal done
        state, payload, pre_truncated = make_request(engine, ex, settings)
        path = cache_dir / f"{cache_key(payload)}.json"
        record = {"id": ex.id, "label": ex.label, "scam_type_label": ex.scam_type, "error": None, "cached": False}
        try:
            if use_cache and path.exists():
                entry = json.loads(path.read_text())
                entry["answers"] = {k: Answer(**a) for k, a in entry["answers"].items()}
                record["cached"] = True
            else:
                async with sem:
                    entry = await _call(engine, state)
                path.write_text(json.dumps(
                    {**entry, "answers": {k: a.model_dump() for k, a in entry["answers"].items()}}
                ))
        except EngineError as exc:
            if exc.status in FATAL_STATUSES:
                raise
            record["error"] = str(exc)
            return record
        finally:
            done += 1
            if done % 100 == 0 or done == len(examples):
                progress(f"  {done}/{len(examples)}")

        record.update(score_answers(entry["answers"], settings))
        record.update({
            "latency_ms": entry["latency_ms"],
            "attempts": entry.get("attempts", 1),
            "input_tokens": (entry.get("usage") or {}).get("input_tokens"),
            "model": entry.get("model"),
            "truncated": pre_truncated or entry.get("truncated", False),
        })
        return record

    return list(await asyncio.gather(*(one(ex) for ex in examples)))


def score_answers(answers: dict[str, Answer], settings: Settings) -> dict:
    """The per-message fields metrics need. `answers` stays in memory only (never written)."""
    v = make_verdict(answers, settings)
    risk = answers.get("risk_level")
    return {
        "p": float(answers["is_scam"].value),
        "verdict": v.verdict.value,
        "scam_type": answers["scam_type"].value if "scam_type" in answers else None,
        "risk_confidence": risk.confidence if risk else None,
        "red_flags": len(v.red_flags),
        "answers": answers,
    }


def fit_from_records(engine: DecisionEngine, records: list[dict]) -> Calibration:
    """Fit temperatures on a calibration split's records (is_scam always; scam_type where labelled)."""
    ok = [r for r in records if r["error"] is None]
    build_payload = getattr(engine, "build_payload", None)
    prompt_version = build_payload({"message": ""}, QUESTIONS)[0].get("prompt_version") if build_payload else None
    return fit_calibration(
        engine.name,
        getattr(engine, "model", None),
        prompt_version,
        [r["answers"] for r in ok],
        {"is_scam": [r["label"] for r in ok], "scam_type": [r["scam_type_label"] for r in ok]},
        QUESTIONS,
    )


def recalibrate(records: list[dict], cal: Calibration, settings: Settings) -> list[dict]:
    out = []
    for r in records:
        if r["error"] is not None:
            out.append(r)
            continue
        out.append({**r, **score_answers(cal.apply(r["answers"], QUESTIONS), settings)})
    return out


# ------------------------------------------------------------------ summary


def summarize(
    records: list[dict],
    *,
    engine_name: str,
    dataset: str,
    split: str,
    settings: Settings,
    limit: int | None,
    target_accuracy: float = 0.9,
) -> dict:
    ok = [r for r in records if r["error"] is None]
    y = np.array([r["label"] for r in ok], dtype=int)
    p = np.array([r["p"] for r in ok], dtype=float)
    latencies = [r["latency_ms"] for r in ok]
    tokens = [r["input_tokens"] for r in ok if r["input_tokens"] is not None]
    models = sorted({r["model"] for r in ok if r.get("model")})
    price = settings.jev_price_per_mtok_usd if engine_name == "jev" else 0.0
    has_data = len(ok) > 0
    return {
        "engine": engine_name,
        "model": models[0] if len(models) == 1 else models,
        "dataset": dataset,
        "split": split,
        "dataset_note": DATASET_NOTES.get(dataset),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "limit": limit,
        "n": len(records),
        "n_scored": len(ok),
        "n_errors": len(records) - len(ok),
        "errors_sample": [r["error"] for r in records if r["error"]][:5],
        "n_positive": int(y.sum()) if has_data else 0,
        "thresholds": {
            "safe": settings.safe_threshold,
            "scam": settings.scam_threshold,
            "min_risk_confidence": settings.min_risk_confidence,
            "red_flag": settings.red_flag_threshold,
        },
        "is_scam": {
            "accuracy_at_0.5": metrics.accuracy(y, p) if has_data else None,
            "roc_auc": metrics.roc_auc(y, p) if has_data else None,
            "ece_10_bins": metrics.ece(y, p) if has_data else None,
            "brier": metrics.brier(y, p) if has_data else None,
            "accuracy_at_80_coverage": metrics.accuracy_at_coverage(y, p, 0.8) if has_data else None,
            "accuracy_at_90_coverage": metrics.accuracy_at_coverage(y, p, 0.9) if has_data else None,
            "recommended_band": metrics.recommend_band(y, p, target_accuracy) if has_data else None,
            "reliability_bins": metrics.reliability_bins(y, p) if has_data else [],
            "coverage_curve": metrics.coverage_curve(y, p) if has_data else [],
        },
        "scam_type": _scam_type_metrics(ok),
        "verdict_policy": metrics.policy_metrics(y, [r["verdict"] for r in ok]) if has_data else None,
        "latency_ms": {
            "p50": metrics.percentile(latencies, 50),
            "p95": metrics.percentile(latencies, 95),
            "note": "Per-check wall time of the original call, including retries; cached reruns reuse it.",
        },
        "cost_per_1000_checks": metrics.cost_per_1000(tokens, price),
        "truncated_count": sum(bool(r.get("truncated")) for r in ok),
        "retried_count": sum(r.get("attempts", 1) > 1 for r in ok),
    }


def _scam_type_metrics(ok: list[dict]) -> dict:
    labelled = [r for r in ok if r.get("scam_type_label")]
    out = {"predicted_counts": _counts(r["scam_type"] for r in ok)}
    if not labelled:
        return {"macro_f1": None, "note": "Not computable: this dataset has no scam_type labels.", **out}
    y_true = [r["scam_type_label"] for r in labelled]
    y_pred = [r["scam_type"] for r in labelled]
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=sorted(set(y_true)), average="macro", zero_division=0)),
        "accuracy": float(np.mean([a == b for a, b in zip(y_true, y_pred)])),
        "n_labelled": len(labelled),
        **out,
    }


def _counts(values) -> dict:
    out: dict = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def write_outputs(summary: dict, records: list[dict], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    date = summary["created_at"][:10]
    prefix = f"{date}-{summary['engine']}-{summary['dataset']}"
    if summary["split"] != "test":  # test keeps the spec's short name; other splits must not overwrite it
        prefix += f"-{summary['split']}"
    if summary["limit"]:
        prefix += f"-limit{summary['limit']}"
    paths = {
        "json": out_dir / f"{prefix}.json",
        "predictions": out_dir / f"{prefix}-predictions.jsonl",
        "reliability": out_dir / f"{prefix}-reliability.png",
        "coverage": out_dir / f"{prefix}-coverage.png",
        "reliability_uncalibrated": out_dir / f"{prefix}-reliability-uncalibrated.png",
    }
    s = summary["is_scam"]
    title = f"{summary['engine']} ({summary['model']}) on {summary['dataset']} {summary['split']}, n={summary['n_scored']}"
    if s["reliability_bins"]:
        charts.reliability_diagram(s["reliability_bins"], s["ece_10_bins"], f"Reliability — {title}", paths["reliability"])
        charts.coverage_chart(s["coverage_curve"], s["accuracy_at_80_coverage"], s["accuracy_at_90_coverage"],
                              f"Accuracy vs coverage — {title}", paths["coverage"])
    summary["charts"] = {k: paths[k].name for k in ("reliability", "coverage")}
    raw = (summary.get("uncalibrated") or {}).get("is_scam")
    if raw and raw["reliability_bins"]:
        charts.reliability_diagram(raw["reliability_bins"], raw["ece_10_bins"],
                                   f"Reliability before calibration — {title}", paths["reliability_uncalibrated"])
        summary["charts"]["reliability_uncalibrated"] = paths["reliability_uncalibrated"].name
    summary["predictions_file"] = paths["predictions"].name
    paths["json"].write_text(json.dumps(summary, indent=2))
    # IDs, labels and outputs only: message text is never written to results.
    paths["predictions"].write_text("".join(
        json.dumps({k: v for k, v in r.items() if k != "answers"}) + "\n" for r in records
    ))
    return paths


# --------------------------------------------------------------------- CLI


def default_engine_factory(name: str, settings: Settings) -> DecisionEngine:
    """The raw engine: caches must hold uncalibrated answers, so calibration can be refit."""
    engine = EngineProvider(settings).get(name)
    return getattr(engine, "inner", engine)


def main(
    argv: list[str] | None = None,
    engine_factory: Callable[[str, Settings], DecisionEngine] = default_engine_factory,
    settings: Settings | None = None,
) -> dict | None:
    parser = argparse.ArgumentParser(prog="python -m second_look.eval.run", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", default="jev")
    parser.add_argument("--dataset", default="sms", choices=sorted(LOADERS))
    parser.add_argument("--split", default="test", choices=["test", "calibration"])
    parser.add_argument("--limit", type=int, default=None, help="only the first N examples of the split")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--no-cache", action="store_true", help="ignore cached responses")
    parser.add_argument("--dry-run", action="store_true", help="print request count and cost; make no calls")
    parser.add_argument("--target-accuracy", type=float, default=0.9)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--calibrate", action="store_true",
                        help="local engines: fit temperatures on the calibration subset, report raw and calibrated")
    parser.add_argument("--calib-size", type=int, default=1_000)
    parser.add_argument("--calibration-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.calibrate and args.split != "test":
        parser.error("--calibrate evaluates on the test split")
    settings = (settings or get_settings()).for_engine(args.engine)

    split_kwargs = {"data_dir": args.data_dir}
    if args.splits_dir:
        split_kwargs["splits_dir"] = args.splits_dir
    examples = load_split(args.dataset, args.split, **split_kwargs)
    if args.limit:
        examples = examples[: args.limit]
    calib_examples: list[Example] = []
    if args.calibrate:
        calib_examples = calibration_subset(load_split(args.dataset, "calibration", **split_kwargs), args.calib_size)

    try:
        engine = engine_factory(args.engine, settings)
    except EngineUnavailable as exc:
        raise SystemExit(f"Engine '{args.engine}' is not available: {exc}") from None
    except KeyError:
        raise SystemExit(f"Unknown engine '{args.engine}'") from None

    cache_dir = args.cache_dir / args.engine
    plan = plan_run(engine, calib_examples + examples, cache_dir if not args.no_cache else Path("/nonexistent"),
                    settings)
    print(
        f"{args.engine} on {args.dataset}/{args.split}: {plan['examples']} examples, "
        f"{plan['cached']} cached, {plan['requests']} new requests, "
        f"~{plan['input_tokens_likely']:,}–{plan['input_tokens_conservative']:,} input tokens, "
        f"est. ${plan['cost_usd_likely']:.4f}–${plan['cost_usd_conservative']:.4f}"
    )
    if args.dry_run:
        return plan

    async def go():
        kw = dict(settings=settings, cache_dir=cache_dir, use_cache=not args.no_cache, concurrency=args.concurrency)
        try:
            calib = await evaluate(engine, calib_examples, **kw) if calib_examples else []
            return calib, await evaluate(engine, examples, **kw)
        finally:
            close = getattr(engine, "aclose", None)
            if close:
                await close()

    try:
        calib_records, records = asyncio.run(go())
    except EngineError as exc:
        raise SystemExit(f"Stopped: {exc}") from None

    common = dict(engine_name=engine.name, dataset=args.dataset, split=args.split, settings=settings,
                  limit=args.limit, target_accuracy=args.target_accuracy)
    cal = None
    if args.calibrate:
        cal = fit_from_records(engine, calib_records)
        raw_summary = summarize(records, **common)
        records = recalibrate(records, cal, settings)
    summary = summarize(records, **common)
    summary["split_hash"] = split_hash(examples)
    if cal is not None:
        summary["uncalibrated"] = {k: raw_summary[k] for k in ("is_scam", "verdict_policy", "scam_type")}
        saved = [cal.save(args.calibration_dir, f"{engine.name}-{args.dataset}")]
        if args.dataset == "sms" and not args.limit:  # the app's default calibration (text messages)
            saved.append(cal.save(args.calibration_dir))
        summary["calibration"] = {**cal.__dict__, "fit_examples": len(calib_records),
                                  "files": [p.name for p in saved]}
    paths = write_outputs(summary, records, args.out_dir)
    s = summary["is_scam"]
    fmt = lambda v, f: "n/a" if v is None else format(v, f)  # noqa: E731
    print(
        f"\nScored {summary['n_scored']}/{summary['n']} (errors: {summary['n_errors']})\n"
        f"  accuracy@0.5   {fmt(s['accuracy_at_0.5'], '.4f')}\n"
        f"  ROC-AUC        {fmt(s['roc_auc'], '.4f')}\n"
        f"  ECE (10 bins)  {fmt(s['ece_10_bins'], '.4f')}\n"
        f"  Brier          {fmt(s['brier'], '.4f')}\n"
        f"  acc@80% cov    {fmt(s['accuracy_at_80_coverage'], '.4f')}\n"
        f"  acc@90% cov    {fmt(s['accuracy_at_90_coverage'], '.4f')}\n"
        f"  latency p50/95 {fmt(summary['latency_ms']['p50'], '.0f')} / {fmt(summary['latency_ms']['p95'], '.0f')} ms\n"
        f"  cost / 1k      ${fmt(summary['cost_per_1000_checks']['usd'], '.4f')}\n"
        + (f"  calibration    T={summary['calibration']['temperatures']} shift={summary['calibration']['shifts']}"
           f" | ECE before "
           f"{fmt(summary['uncalibrated']['is_scam']['ece_10_bins'], '.4f')} -> after {fmt(s['ece_10_bins'], '.4f')}\n"
           if cal is not None else "") +
        f"Wrote {paths['json']}"
    )
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
