# PLAN — Second Look

- **Phase 1** (Decision layer + Jev): **done** 2026-10-10. See the Phase 1 sections below.
- **Phase 2** (open-source engines + calibration): **draft, awaiting approval.** See [Phase 2](#phase-2--open-source-engines--calibration) at the end.

# Phase 1 — Decision layer + Jev


---

## 1. Jev API check (docs.typesafe.ai, read 2026-10-09)

Sources: `docs.typesafe.ai/api.md`, `/models.md`, `/introduction/quickstart.md`, `/primitives.md`, `/primitives/score.md`, `/model-jaggedness/jev-1.13.md`, `/sdk/python/api/retries.md`.

| Item | Docs say | Spec says | Impact |
|---|---|---|---|
| Endpoint | `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <key>` | Same | None |
| Request | `{state, model, questions}`. `state` may be a string, object or array. | Same; `state` is a JSON object | None |
| Noul question | `type`, `instructions`, and optional `criteria: {"true": …, "false": …}` | No criteria | We can leave it out. Optional improvement. |
| Choice question | `criteria: map<option, description \| null>`, max 255 options | `dict[str, str]` | None |
| Score question | `criteria: [levels low→high]`, 2–10 levels | Same | None |
| **Noul answer** | `{"type":"noul","noul":0.93}`. **No `probabilities` and no `confidence`.** | Answer has `probabilities` and `confidence` | The mapper builds `probabilities={"true":p,"false":1-p}` and sets `confidence=None`. |
| Choice answer | `{"choice", "probabilities", "confidence"}` | Matches the Answer model | The mapper renames `choice` to `value`. |
| Score answer | `{"score", "legend", "probabilities" (keys "0".."n-1"), "confidence"}`. Score = probability-weighted level index, range 0..n-1. | "expected level" | The mapper renames `score` to `value` and keeps the string level keys. |
| Score `confidence` | Computed from how spread out the probabilities are; 1.0 when all mass is on one level. The exact formula is not documented. | §5 defines local confidence as top probability minus the second-highest | The `risk_level.confidence < 0.4` threshold may not mean the same thing across engines. This matters in Phase 2, not now. |
| Model | Only `jev-1.13.0` exists. `jev-latest` and `jev-preview` both point to it. The docs recommend pinning the versioned ID. | Pin e.g. `jev-1.13.0` | Pin `jev-1.13.0` in config. |
| Limits | 64K tokens per request; 32K for state plus the longest question; 80 req/s and 100K tok/s. No documented maximum on the number of questions. | 64K / 32K | No tokenizer is published, so we estimate tokens conservatively (chars ÷ 3) and middle-truncate `message` to fit. |
| Errors | 401, 422, 429, **529 Overloaded**. The SDK also retries 408 and 5xx. Honor `Retry-After` / `retry-after-ms`. 413 is not documented. | "429/5xx" | Retry on 408, 429 and 5xx (529 included), honoring `Retry-After`. Don't retry other 4xx; raise `EngineError` instead. |
| Pricing | **$0.042 per 1M input tokens**; output is free; no per-request fee | "input-token pricing" | Cost per check = `usage.input_tokens` × 0.042e-6, taken from the response. |
| Response extras | `model` (resolved version) and `usage.{input_tokens,output_tokens}` | Not mentioned | Store both in `Answer.raw` and in the eval output. |
| Python SDK | `typesafe-sdk` exists | Spec uses httpx | We use **httpx as the spec says** and don't add the SDK. |
| Known weak spots (jaggedness page) | Adversarial state ("text that argues for its own classification") can move the answer. Bias toward the **first Choice option**. Long irrelevant state hurts. | — | Put `not_a_scam` / `none` options **last** in choice criteria. Flag adversarial scam text as a known risk in eval notes. |

---

## 2. Layout

The repo root (this folder) is the spec's `second-look/`:

```
spec.md  PLAN.md  README.md  .gitignore  .env.example
api/
  pyproject.toml  uv.lock
  second_look/
    __init__.py  config.py  main.py  questions.py  preprocess.py  verdict.py  storage.py
    engines/  __init__.py  base.py  fake.py  jev.py  registry.py
    eval/     __init__.py  download.py  datasets.py  metrics.py  charts.py  run.py
              data/      (gitignored)
              cache/     (gitignored; Jev responses by request hash)
              results/   (committed: <date>-<engine>-<dataset>.json + PNGs)
  tests/
```

The folder isn't a git repository yet. Step 1 runs `git init`.

---

## 3. Build order, modules and tests

Each step ends with a green `uv run pytest`, one commit, and a 2–3 line summary.

### Step 1: Scaffolding
- uv project for Python 3.12. `main.py` has a FastAPI app with `GET /api/health` returning `{"status":"ok"}`.
- `config.py`: settings come from environment variables with typed defaults: thresholds (0.25 / 0.75 / 0.4 / 0.6), model ID, timeout, retries, DB path and `TYPESAFE_API_KEY`.
- Also: `.env.example` (placeholder only), `.gitignore` (`.env`, `.venv`, `eval/data`, `eval/cache`, `*.db`), and a README with "How to run".
- **Tests:** `/api/health` returns 200.

### Step 2: Engine interface (`engines/base.py`, `engines/fake.py`)
- `Noul`, `Choice`, `Score`, `Answer` and `DecisionEngine` exactly as in §5. Add `EngineError` and `EngineUnavailable`.
- `FakeEngine(name, scripted: dict[str, Answer] | Callable)` plus small helpers: `noul(p)`, `choice(opt, probs, conf)`, `score(levels, probs)`.
- **Tests:** models validate and reject bad `type` values. `FakeEngine` satisfies the protocol and returns its scripted answers. A score helper computes the expected level correctly.

### Step 3: Question set (`questions.py`)
- `QUESTIONS: dict[str, Noul|Choice|Score]` with the nine keys from §6.
- `scam_type` has a one-line description per option. `payment_method` and `impersonation` get short descriptions too. The "none" option goes last.
- Also exports `FLAG_KEYS` (questions that drive red flags).
- **Tests:** exactly the nine §6 keys and types. Choice options match §6 exactly. `risk_level` levels are in order. Every instruction is one sentence, and no option description contains " and " / " or " compound decisions (lint-style check).

### Step 4: Preprocessing (`preprocess.py`)
- `build_state(text, channel, claimed_sender) -> (state: dict, meta: PreprocessMeta)`.
- `meta` holds `truncated`, `original_chars` and `estimated_tokens`.
- Normalization: Unicode NFKC, strip zero-width characters, collapse whitespace, de-obfuscate `hxxp` and `[.]` in URLs.
- URL extraction: scheme or bare-domain URLs. Domains come from **tldextract run offline** (bundled suffix snapshot, no network). We record `registered_domain` and keep look-alike and punycode forms visible (`xn--` decoded alongside).
- Phone extraction: US/intl formats and 5–6 digit short codes in a plain "text" context.
- Amount extraction: `$1,234.56`, `$5`, `USD 20`, `20 dollars`, `1.5k`, `€`/`£` prefixed amounts.
- `truncate_middle(text, max_tokens)`: keeps the head and tail and inserts a `[… N characters removed …]` marker.
- **Tests (tricky cases):**
  - shortened URLs (`bit.ly/x`, `t.co/…`, no scheme)
  - look-alikes (`paypa1.com`, `usps.com.track-fee.top`, `amaz0n-support.co.uk`, punycode `xn--pypal-4ve.com`)
  - URLs with trailing punctuation
  - phone numbers: `(555) 123-4567`, `+1 555.123.4567`, `+44 20 7946 0958`, short code `7726`
  - amounts: `$3.99`, `$1,250`, `USD 50`, `50 dollars`, `$1.2k`, `£20`
  - numbers that are not phones or amounts (dates, order IDs)
  - truncation fires only over the limit; head and tail are preserved; the flag is set

### Step 5: Jev engine (`engines/jev.py`)
- `JevEngine(api_key, model="jev-1.13.0", timeout=5.0, max_retries=3)` wraps an `httpx.AsyncClient`.
- `decide()` sends **one POST** with all questions and maps each answer type to `Answer`. Raw JSON is kept in `raw`, and the resolved model plus usage are exposed.
- Retries on 408, 429 and 5xx use exponential backoff with jitter (0.5 s doubling, capped) and honor `Retry-After` / `retry-after-ms`. Other 4xx raise `EngineError` with the status and request ID.
- Before sending, the engine checks the token budget and middle-truncates if needed (it uses the step-4 helper, so truncation is recorded).
- **Tests (respx, real network blocked by a conftest fixture):**
  - success maps all three answer types
  - 429 then 200 succeeds, with `Retry-After` honored (sleep patched)
  - 529 then 200
  - timeout on every attempt raises `EngineError`
  - 401 and 422 raise immediately with no retry
  - an over-limit message is sent truncated, the flag is set, and the payload is under 32K estimated tokens
  - the request body has `model == "jev-1.13.0"` and the Bearer header

### Step 6: Verdict logic (`verdict.py`)
- `make_verdict(answers, settings) -> Verdict{verdict, probability, scam_type, red_flags[], next_steps[]}`.
- Pure functions; imports nothing from `engines/` except the types.
- Abstain rule: `low < p < high` or `risk_level.confidence < min_conf` gives `UNSURE`.
- `p >= high` gives `LIKELY_SCAM`. `p <= low` gives `PROBABLY_SAFE`.
- Red flags: Noul flags at P(true) ≥ 0.6. Choice flags are covered in Q1 below.
- Next steps: a table keyed by `scam_type`, plus the always-on block (don't click, don't reply, reportfraud.ftc.gov, forward to 7726).
- `PROBABLY_SAFE` adds "can be wrong / verify independently". `UNSURE` adds "here's how to check".
- **Tests (FakeEngine):**
  - p = 0.25 → SAFE, 0.2501 → UNSURE, 0.75 → SCAM, 0.7499 → UNSURE
  - high p with low risk confidence → UNSURE
  - risk confidence exactly 0.4 → not abstained
  - flag at 0.6 shown, 0.5999 hidden
  - every `scam_type` has next steps
  - reporting steps always present
  - the SAFE caveat is present
  - thresholds are overridable via config
  - every output string is a member of the template tables (proves there's no generated text)

### Step 7: API
- `POST /api/check`:
  - takes `{text, channel, claimed_sender?, engine?}`
  - returns the §8 response plus `check_id` (see Q3)
  - `image` returns 422 "screenshots arrive in Phase 3"
- `GET /api/engines` returns `[{name:"jev", ready, reason}]`.
- With no key: 503 `{"detail":"No decision engine is configured: set TYPESAFE_API_KEY …"}`.
- **Tests (TestClient with an injected FakeEngine):**
  - happy path
  - 503 with no key
  - validation errors
  - unknown engine gives 400
  - engine failure gives 502 with a template message

### Step 8: Storage (`storage.py`) + feedback
- SQLModel tables:
  - `Check(id, created_at, verdict, probability, engine, model, latency_ms, truncated, stored_text?)`
  - `Feedback(id, check_id, correct, true_label?)`
- `help_improve: bool = false` on the request. Text is saved only when it's true, with phones and emails masked (`[phone]`, `[email]`).
- `POST /api/checks/{id}/feedback`: 404 for an unknown ID.
- **Tests:**
  - by default no text is stored (assert on the DB row)
  - with opt-in, text is stored masked
  - feedback round trip
  - feedback on an unknown ID gives 404

### Step 9: Eval harness (`eval/`)
- `download.py` fetches the UCI SMS Spam Collection zip into `eval/data/`, checks its sha256, and is gitignored.
- `datasets.py` loads the data and makes a **stratified 60/40 split with seed 42**. The split IDs are written to `eval/splits/sms.json` (committed, so the split is fixed).
- `run.py`:
  - CLI: `python -m second_look.eval.run --engine jev --dataset sms [--split test] [--limit N] [--concurrency 8] [--no-cache]`
  - disk cache at `eval/cache/<sha256(model+state+questions)>.json`
- `metrics.py` computes for `is_scam`:
  - accuracy
  - ROC-AUC
  - ECE (10 bins)
  - Brier score
  - accuracy-vs-coverage curve (abstain band widened symmetrically around 0.5), with accuracy at 80% and 90% coverage
  - p50/p95 latency (measured on uncached calls only)
  - cost per 1,000 checks from actual `usage.input_tokens`
- Also reports how often `UNSURE` comes up at the current thresholds.
- `charts.py` draws a reliability diagram and a coverage curve (matplotlib, Agg backend). Output: `eval/results/<date>-jev-sms.json` plus PNGs.
- **Tests (synthetic data, FakeEngine):**
  - metrics against hand-computed values (perfectly calibrated gives ECE ≈ 0; known Brier and AUC)
  - the coverage curve is monotone
  - the split is deterministic and stratified
  - a cache hit avoids a second engine call
  - `--limit` works
  - the end-to-end CLI with a fake engine writes JSON and two PNGs

### Before the first real Jev call
I stop and report the request count and the estimated cost, then wait for your OK.

Rough estimate now:
- The SMS test split is about 40% of 5,574, so about **2,230 requests**.
- Each request is about 700–1,000 input tokens (state plus 9 questions with criteria).
- That's about 2.2M tokens × $0.042/M ≈ **$0.10**.

I'll suggest a `--limit 50` trial first.

---

## 4. Open questions (need your call)

**Q1. Red flags from Choice questions.**
§7 says "every flag question whose probability ≥ 0.6", but `payment_method` and `impersonation` are Choices.
- *Proposal:* a flag fires when any option other than `none_requested` / `none` has probability ≥ 0.6, with one template per option (e.g. `gift_card` → "Asks for payment in gift cards — no real company or agency does this.").
- The red-flag set would be: urgency, asks_credentials, link_mismatch, too_good, payment_method (non-none), impersonation (non-none).
- `is_scam`, `scam_type` and `risk_level` are not flags.

**Q2. SMS "spam" ≠ "scam".**
UCI labels marketing spam as `spam` too.
- *Proposal:* treat `spam` as the positive label for `is_scam` in this dataset and say so in the results JSON. Jev may correctly call legit-but-spammy marketing "not a scam", which will look like errors. We'll report it rather than redefine the question.
- `scam_type` macro-F1 is **not computable** on SMS (no type labels), so it's reported as `null` until Phase 2's hand-labeled set.

**Q3. Feedback needs an ID.**
The §8 `/api/check` response has no ID, but `/api/checks/{id}/feedback` needs one.
- *Proposal:* add `check_id` to the response.
- Also add the opt-in field `help_improve: bool` to the request. §8 implies it but doesn't list it.

**Q4. Dependencies not literally in §3.** I'm asking before adding:
- `uvicorn` (to run FastAPI)
- `tldextract` (named in §6 and your brief)
- `numpy` (comes with pandas and sklearn anyway)
- dev only: `pytest-asyncio` (async engine tests)
- I'd **not** add `python-dotenv` or `pydantic-settings`. Config reads `os.environ` directly, and the README says `export TYPESAFE_API_KEY=…` or `uv run --env-file .env`.

**Q5. Timeout semantics.**
- *Proposal:* 5 s is **per attempt**, with at most 3 retries. The total wall time is capped at about 15 s, and the eval logs latency per attempt plus total.
- The other reading is 5 s total including retries.

**Q6. Abstain threshold tuning.**
§7 says thresholds are "tuned from the coverage curve".
- *Proposal:* Phase 1 ships the §7 defaults in config. The eval JSON reports which band would hit 90% accuracy at the highest coverage, as a *recommendation*. I won't change config automatically.

**Q7. `risk_level.confidence` for Jev** is Jev's own (undocumented) spread measure, not "top minus second".
- *Proposal:* use it as-is for Jev in Phase 1, and note it for Phase 2 cross-engine comparison.

---

# Phase 2 — Open-source engines + calibration

Scope: spec §11 Phase 2. That covers:
- the LLM option-scoring engine (A)
- the NLI engine (B)
- temperature scaling
- the phishing-email dataset
- the hand-labelled recent-scam set
- a three-way engine comparison and `RESULTS.md`

Out of scope: the web client, OCR and Docker (Phase 3).

Status: **draft, awaiting approval.** Decisions D1–D7 are in §P5.

## P1. What I checked (2026-10-10)

**Machine:** Apple M1 Pro, 16 GB unified memory, 622 GB free disk, no Hugging Face cache yet. Models run on the GPU via PyTorch's MPS backend. A 4B model in 16-bit takes about 8 GB, which fits next to everything else, but not with much room to spare.

**Models (Hugging Face, read today):**

| Use | Model | License | Weights | Notes |
|---|---|---|---|---|
| Engine A (recommended) | `Qwen/Qwen3-4B-Instruct-2507` | Apache-2.0 | 8.06 GB | Newer update of Qwen3-4B's non-thinking mode; non-thinking only, so no `<think>` tokens interfere with scoring. Card benchmarks well above Qwen3-4B non-thinking (MMLU-Pro 69.6 vs 58.0). Needs transformers ≥ 4.51. |
| Engine A (fallback) | `Qwen/Qwen3-1.7B` | Apache-2.0 | 4.08 GB | Use it if the 4B model is too slow or runs out of memory. Needs `enable_thinking=False`. |
| Engine B (recommended) | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c` | MIT | ~0.87 GB | The "-c" variant was trained only on commercially usable data, and its card scores it higher on spam (f1 0.74 vs 0.60 for the non-c model). **I still need to confirm the repo exists before downloading.** |
| Engine B (alt.) | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` | MIT* | 870 MB | *Trained partly on non-commercially licensed NLI data, which the card flags. |

The NLI models accept at most 512 tokens, so engine B middle-truncates long emails using the same helper as Jev.

**Phishing-email datasets:**

| Dataset | License | Size | Problems |
|---|---|---|---|
| `zefang-liu/phishing-email-dataset` (recommended) | LGPL-3.0 | 18,650 emails, CSV 52 MB | A copy of a Kaggle set (Enron + list mail). "Phishing" includes generic spam. Has null/empty rows, Enron-style lowercased text, and some rows up to ~17M characters. |
| `ealvaradob/phishing-dataset` | Apache-2.0 | 1.7 GB, mixed | Same email source, mixed with URLs, SMS and HTML, and loading it needs `trust_remote_code`. Not recommended. |
| `cybersectony/PhishingEmailDetectionv2.0` | **none declared** | 22.6k emails + 177k URLs | Unclear terms. Not recommended. |

All three draw on the same 2000s-era Kaggle and Enron emails, which is why the hand-labelled recent set matters.

**Libraries:** torch 2.14.1 (macOS arm64 wheels) and transformers 5.19.0, which needs PyTorch ≥ 2.5. scipy is already listed in spec §3.

## P2. The runtime problem, and how I'd handle it

The spec asks for a local check in under 3 s (§2). Done the obvious way, engine A runs 9 prompts per message, each repeating the message, which is roughly 5,000 tokens per check. That could take 5–15 s on an M1 Pro.

The plan:
- **Reuse the message across questions.** Encode the system prompt and message once, cache that work, then score all 9 question suffixes in a single batch. That brings it to roughly 400 + 9 × 150 tokens per check.
- **Measure before committing.** Benchmark on 20 messages first. If it's still over about 3 s per check, fall back to Qwen3-1.7B (D2).
- **Eval runtime:** at about 2–3 s per message, the 2,064 test messages take about 1.5 h for engine A and a few minutes for engine B. Runs go in the background and use the same on-disk cache, so they can be stopped and resumed.
- **Calibration fit:** fit the temperatures on a fixed **1,000-message stratified subsample** of the calibration split (D3), not all 3,096. Temperature scaling fits one number per question, and 1,000 messages is plenty for that. This saves about 1.5 h per local engine.

## P3. Build order (one commit per step, `pytest` green each time)

Every step ends with a green `pytest`, one commit and a 2–3 line summary.

**Step 10: dependencies + device helper.**
- Add `torch` and `transformers` (D1).
- Add `sentencepiece` or `protobuf` only if DeBERTa's tokenizer actually needs them; I'll ask first if so.
- `engines/device.py` picks MPS, then CUDA, then CPU, and handles dtype (bf16 on MPS, falling back to fp16).
- Model weights download to the standard HF cache, never into the repo.
- **Tests:** device selection logic, with torch availability mocked.

**Step 11: engine A, LLM option scoring (`engines/llm_scoring.py`).**
- The state dict renders to a fixed text block: message, channel, claimed sender, URLs with registered domains, phones, amounts.
- Each question gets a prompt with its instructions and lettered options (A, B, C…). We read the next-token log-probabilities for the option letters only and softmax them into probabilities.
- Per type, as spec §5 describes:
  - Noul: 2 options, value = P(true).
  - Choice: value = the top option.
  - Score: value = the probability-weighted level index.
  - Confidence = top probability minus the second.
- Shared-prefix cache plus one batched forward pass for all 9 questions.
- The forward pass is isolated behind a `Scorer` interface, so tests never load real weights.
- To counter first-option bias, the order of options stays as in `questions.py` (none last). I'll check on a small set whether letter position bias is a problem.
- **Tests:**
  - prompt rendering is deterministic, and every option gets exactly one letter token
  - logits → probabilities and Answer mapping for each question type, using a fake scorer with hand-set logits
  - the score expected-level maths
  - confidence definition
  - letter tokens are single tokens for the real tokenizer
  - truncation of long messages
- Optional `@pytest.mark.slow` test runs the real model on 2 messages; it's skipped by default.

**Step 12: engine B, NLI (`engines/nli.py`).**
- Each option becomes a hypothesis, e.g. "This message is a package-delivery scam."
- The premise is the rendered state, middle-truncated to fit 512 tokens.
- Per type:
  - Noul: entailment vs contradiction of the instruction itself.
  - Choice and score: softmax over each option's entailment logit.
- All option/hypothesis pairs for one message go in one batch.
- I'll write the hypothesis templates per option in `questions.py`, keeping it the single source of truth (e.g. `criteria` descriptions become hypotheses).
- **Tests:** hypothesis construction for every question and option, logits → Answer for each type with a fake NLI scorer, and the 512-token truncation.

**Step 13: registry + API.**
- `EngineProvider` gains `llm` and `nli`. `GET /api/engines` reports `ready` from whether weights are in the local HF cache, and the endpoint never triggers a download.
- `POST /api/check` with `engine: "llm"` or `"nli"` works.
- Phase 3 needs this fallback order, set up now:
  - default to Jev when its key is present
  - otherwise use the best ready local engine, and say so in the response
- **Tests:** readiness with a fake cache, engine selection, and the 503 only when *no* engine is ready.

**Step 14: calibration (`calibration.py`).**
- One temperature per question. Noul: sigmoid(logit/T). Choice and score: softmax(logits/T).
- Fit by minimising NLL with `scipy.optimize.minimize_scalar` over log T. Store as JSON per engine and model version in `eval/calibration/`, apply at inference, and record it in `Answer.raw`.
- **Only questions with labels can be calibrated (D4).**
  - `is_scam` gets labels from SMS, phishing and hand-labelled data.
  - `scam_type` gets them only from the hand-labelled set.
  - The other seven questions have no labels anywhere, so they keep T = 1 and the report says so.
- **Tests:**
  - T recovered on synthetic over- and under-confident data
  - T = 1 leaves probabilities unchanged
  - apply/fit round trip
  - ECE drops on synthetic miscalibrated data
  - the calibration file is versioned by model ID

**Step 15: datasets.**
- `phishing` loader:
  - download with a pinned sha256
  - drop null/empty rows and exact duplicates
  - drop rows over 50k characters; the spec's limits make them unrepresentative anyway
  - fixed seeded **2,000-email stratified subsample** (D5), then the same 60/40 split, committed
- `handlabeled` loader plus `python -m second_look.eval.handlabeled validate <file>`:
  - checks the schema (text, label, `scam_type` from `questions.py`, channel, received month, source)
  - runs `mask_pii`
  - flags anything that still looks like a phone number, email, long digit run or `@handle` for manual review
- The data file stays **out of git by default** (D6).
- **Tests:** loader cleaning rules, deterministic subsample and split, and every validator rule.

**Step 16: harness and comparison.**
- `eval.run --engine llm|nli` runs on the same splits. It gains `--calibrate`, which fits temperatures on the calibration split and evaluates raw and calibrated predictions on test in one go, both in the results JSON.
- `scam_type` macro-F1 is filled in when labels exist.
- New `eval.compare` writes `eval/results/<date>-comparison.json` and a combined reliability chart, one panel per engine, so no chart has two y-axes.
- `RESULTS.md` covers, with numbers, where the local engines match Jev and where they don't. That includes raw vs calibrated ECE (the Phase 2 criterion), accuracy at coverage, latency on this M1 Pro and $0 cost.
- The threshold sweep re-runs per engine: thresholds may differ per engine in config, keyed by engine name.
- **Tests:** compare output with fake results, and `--calibrate` end to end with FakeEngine on synthetic data.

**Before any long run or paid call:** I stop and report the expected requests, cost and time:
- local runs: time estimate from the 20-message benchmark
- Jev on phishing and hand-labelled data: dry-run cost estimate

## P4. Phase 2 acceptance criteria (spec §11)

| Criterion | How it will be shown |
|---|---|
| All three engines run through the same harness on the same test split | `eval.run --engine jev|llm|nli --dataset sms`, same committed split file; comparison JSON lists the shared split hash |
| Calibrated local engine's ECE lower than uncalibrated on test | Both numbers in each local engine's results JSON and in `RESULTS.md`, with before/after reliability charts |
| `RESULTS.md` states, with numbers, where the open-source engine matches Jev and where it doesn't | Written from the comparison JSON; per dataset, per metric |

## P5. Decisions needed

- **D1, dependencies.** Add `torch`, which the spec doesn't list but `transformers` needs to run models. Also allow `sentencepiece` or `protobuf` if DeBERTa's tokenizer turns out to need them; I'll confirm that before adding either. `accelerate` isn't needed because the models load onto one device.
- **D2, engine A model.** `Qwen3-4B-Instruct-2507` (recommended), with automatic fallback to `Qwen3-1.7B` if it's over about 3 s per check after the caching optimisation. The 4B model is an 8 GB download.
- **D3, calibration fit set.** A 1,000-message stratified subsample of the calibration split (recommended), or the full 3,096, which adds about 1.5 h per local engine.
- **D4, calibration scope.** Calibrate only questions with labels (`is_scam` and, once hand-labelled data exists, `scam_type`), and report the others as uncalibrated. Spec §5 says "per question", but there's no data to do that for the other seven.
- **D5, phishing data.** `zefang-liu/phishing-email-dataset` (LGPL-3.0), cleaned, with a fixed 2,000-email subsample.
  - That's 800 test emails. Running Jev on them costs about **$0.05–0.10**, confirmed by dry run before spending.
  - The full set would be about 7k test emails: roughly $0.75 for Jev and many hours locally.
- **D6, hand-labelled set.** I'll build the format, validator and PII checks; **you supply the messages.**
  - Aim for 150–300 recent scams *plus* about 100 real legitimate messages (bank alerts, delivery updates, 2FA codes). Without legitimate ones, the set can't measure false alarms.
  - Keep the file out of git (recommended), or commit it after the PII checks pass.
- **D7, order.** Build engines and calibration on SMS and phishing first, and add the hand-labelled set whenever it's ready. The comparison re-runs at no cost from the caches.
