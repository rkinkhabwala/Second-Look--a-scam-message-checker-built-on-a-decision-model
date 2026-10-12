# Second Look — a scam-message checker built on a decision model

> Working name: **Second Look**. Paste (or screenshot) a suspicious text, email, or DM and get back: how likely it is to be a scam, what kind of scam, which red flags are present, and what to do next. The decisions come from **Jev** (TypeSafe's "System One" decision model) or from an **open-source decision engine** that runs locally — the app can switch between them and benchmark them against each other.

---

## 1. The problem

Scam texts and emails (fake toll bills, package-delivery fees, bank-impersonation, job offers, romance and crypto "investment" pitches) cost people billions a year, and the people most often hit — older adults, recent immigrants, first-time job seekers — are the least likely to recognize them. Existing filters are binary and silent: they either hide a message or don't, and never explain why.

Second Look gives a **calibrated** answer ("87% likely a scam") plus the **specific red flags** found, so the user learns the pattern, and it **says when it isn't sure** instead of guessing.

### Why a decision model fits
Every judgment here is multiple-choice, not writing: *Is it a scam? Which type? Does it create urgency? Does it ask for payment in gift cards or crypto? Does it impersonate a known organization?* A decision model answers all of these in one pass, with a probability on each, which lets the app:
- act on high-confidence answers and show "Not sure — here's how to check" on low-confidence ones (selective prediction);
- show red flags as checklist items driven by probabilities, with no free-text generation and so nothing hallucinated.

---

## 2. Goals and non-goals

**Goals**
- Check a message in under 1 second with Jev, under 3 seconds locally.
- Pluggable decision engines behind one interface: Jev, and at least two open-source engines.
- A reproducible evaluation harness comparing engines on accuracy, calibration, latency, and cost.
- A clean web app a non-technical person can use.

**Non-goals (v1)**
- Reading the user's inbox or SMS automatically, browser/phone extensions, blocking messages, accounts/multi-user, cloud deployment.
- Visiting links in messages. URLs are analyzed as text only; the app never fetches them.

---

## 3. Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.12, `uv` for env/deps |
| API | FastAPI, Pydantic v2 |
| Jev engine | `httpx` client against `POST https://api.typesafe.ai/v1/systemone` (model pinned, e.g. `jev-1.13.0`) |
| Open-source engine A | **LLM option-scoring**: Qwen3-4B (or similar open-weight model) via Hugging Face `transformers`, scoring each option's log-likelihood → softmax → probabilities |
| Open-source engine B | **Zero-shot NLI classifier**: a DeBERTa-v3 zero-shot model (`transformers` pipeline) — fast CPU baseline |
| Calibration | Temperature scaling per question, fit on a held-out split (`scipy`) |
| OCR (screenshots) | Tesseract via `pytesseract` |
| Storage | SQLite (checks, user feedback, eval runs) via SQLModel |
| Web client | React + TypeScript + Vite, Tailwind |
| Eval / charts | pandas, scikit-learn metrics, matplotlib; results also exported as JSON for the UI |
| Testing | pytest, respx (mock Jev HTTP), Vitest |
| Local infra | Docker Compose (api, web); model weights cached in a volume |

Ask before adding anything not listed.

---

## 4. Repository layout

```
second-look/
├── spec.md
├── docker-compose.yml
├── api/
│   ├── pyproject.toml
│   └── second_look/
│       ├── main.py               # FastAPI app
│       ├── questions.py          # the question set (single source of truth)
│       ├── engines/
│       │   ├── base.py           # DecisionEngine protocol + typed answers
│       │   ├── jev.py
│       │   ├── llm_scoring.py    # open-source engine A
│       │   └── nli.py            # open-source engine B
│       ├── calibration.py
│       ├── verdict.py            # turns answers into the user-facing result
│       ├── preprocess.py         # normalize text, extract URLs/phones/amounts, OCR
│       ├── storage.py
│       └── eval/
│           ├── datasets.py
│           ├── run.py            # CLI: python -m second_look.eval.run --engine jev
│           └── metrics.py
│   └── tests/
└── web/
    └── src/ (pages: Check, Result, Learn, Benchmarks, Settings)
```

---

## 5. The decision engine interface

Mirror Jev's three question types so every engine is interchangeable:

```python
class Noul(BaseModel):        # yes/no → P(true)
    type: Literal["noul"] = "noul"
    instructions: str

class Choice(BaseModel):      # one of N options
    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str]  # option -> when it applies

class Score(BaseModel):       # ordered scale
    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str]       # lowest -> highest

class Answer(BaseModel):
    type: str
    value: str | float        # chosen option, P(true), or expected level
    probabilities: dict[str, float]
    confidence: float | None
    raw: dict | None = None

class DecisionEngine(Protocol):
    name: str
    async def decide(self, state: str | dict, questions: dict[str, Noul | Choice | Score]) -> dict[str, Answer]: ...
```

**Jev engine:** sends `{model, state, questions}` in one request and maps `answers` back. Respect Jev's limits (64K tokens per request, 32K for state + longest question); truncate long emails from the middle and record that a truncation happened. Retries with backoff on 429/5xx; timeout 5 s.

**LLM option-scoring engine (open-source Jev alternative):**
- For each question, build a prompt with the message, the instructions, and lettered options; compute the log-probability of each option's letter token in one forward pass; softmax → probabilities.
- `noul` = two options (true/false). `score` = levels as options; value = probability-weighted level index (same definition Jev uses).
- `confidence` = top probability minus second-highest.
- Batch all questions for one message in a single batched forward pass.

**NLI engine:** each option becomes a hypothesis ("This message is a package-delivery scam."); entailment scores → softmax across options. Fastest, weakest; it's the baseline.

**Calibration:** raw probabilities from the open-source engines are not calibrated out of the box. Fit one temperature per question on a validation split and apply it at inference. Jev is used as-is (it is sold as calibrated — the eval harness checks that claim rather than assuming it).
  - *Amended 2026-10-12:* `noul` questions use **Platt scaling** (`sigmoid(z/T + shift)`): a temperature plus one shift per question. On 400 held-out SMS messages (calibration split, outside the fit subset), temperature alone made Qwen3-1.7B's ECE worse (0.180 → 0.207, 35 false alarms unchanged) because the model leans towards "yes"; temperature + shift gave ECE 0.059 and accuracy 0.960 (from 0.820). `choice` and `score` questions keep plain temperature scaling.

---

## 6. The question set (`questions.py`)

One atomic decision per question (Jev reads questions literally; compound questions hurt accuracy).

| key | type | instructions / criteria |
|---|---|---|
| `is_scam` | noul | "This message is a scam or fraud attempt." |
| `scam_type` | choice | `delivery_fee`, `toll_or_fine`, `bank_or_card`, `government_impersonation`, `tech_support`, `job_offer`, `romance`, `investment_crypto`, `prize_or_lottery`, `account_verification`, `family_emergency`, `other_scam`, `not_a_scam` — each with a one-line description |
| `urgency` | noul | "The message pressures the reader to act immediately or face a penalty." |
| `payment_method` | choice | `gift_card`, `crypto`, `wire_or_zelle`, `card_on_link`, `none_requested` |
| `asks_credentials` | noul | "The message asks for a password, PIN, one-time code, or account number." |
| `impersonation` | choice | `bank`, `government_agency`, `delivery_company`, `tech_company`, `employer_or_recruiter`, `family_member`, `none` |
| `link_mismatch` | noul | state includes extracted URLs and claimed sender → "A link's domain does not match the organization the message claims to be from." |
| `too_good` | noul | "The message offers money, prizes, or returns that are unrealistically generous." |
| `risk_level` | score | `["safe", "probably safe", "suspicious", "likely scam", "scam"]` |

The **state** sent to every engine is a JSON object, not raw text: `{ "message": ..., "channel": "sms|email|dm", "claimed_sender": ..., "urls": [...], "phone_numbers": [...], "amounts": [...] }`, built by `preprocess.py` with plain regex/`tldextract`. Deterministic features are extracted in code, never asked of the model.

---

## 7. Turning answers into a verdict (`verdict.py`)

Pure, deterministic, unit-tested — no model calls.

- `p = is_scam.value` (after calibration for local engines).
- **Abstain band:** if `0.25 < p < 0.75` **or** `risk_level.confidence < 0.4` → verdict `UNSURE`. Thresholds live in config and are tuned from the coverage/accuracy curve in §9, not guessed.
  - *Tuned 2026-10-10 (Jev `jev-1.13.0`, tuned on the SMS calibration split, checked on test):* `safe ≤ 0.25`, `scam ≥ 0.85`, and the `risk_level.confidence` gate **off** (`min_risk_confidence = 0`). On the test split this answers 84.4% of messages at 99.37% accuracy (was 76.7% at 99.18%), with 6 of 257 scams called safe (was 5). The gate cost coverage without improving accuracy: 198 of the 199 messages it alone held back would have been answered correctly. The gate stays in config, so it can be re-enabled per engine. Re-tune once the hand-labelled recent-scam set exists (Phase 2).
- `p >= 0.75` → `LIKELY_SCAM`; `p <= 0.25` → `PROBABLY_SAFE`.
- **Red flags** = every flag question whose probability ≥ 0.6, rendered from fixed, human-written templates (e.g. *"Asks for payment in gift cards — no real company or agency does this."*).
- **Next steps** come from a fixed table keyed by `scam_type` (e.g. toll scams → "Check your balance only on your state's official toll website — type the address yourself"; family emergency → "Call the person on a number you already have"). Always include: don't click links, don't reply, how to report (reportfraud.ftc.gov, forward texts to 7726).
- A `PROBABLY_SAFE` result always says it can be wrong and how to verify independently.

No generated text anywhere in the result — every sentence the user sees is a template you wrote.

---

## 8. API

- `POST /api/check` — `{ text?, image? (base64), channel, claimed_sender?, engine? }` → `{ verdict, probability, scam_type, red_flags[], next_steps[], engine, latency_ms, truncated }`
- `POST /api/checks/{id}/feedback` — `{ correct: bool, true_label? }` (user says "this was actually safe/a scam")
- `GET /api/engines` — available engines and whether each is ready (key present, weights downloaded)
- `GET /api/benchmarks` — latest eval results per engine
- `GET /api/health`

Privacy: messages are stored only if the user ticks "help improve" on a result; otherwise only the verdict, engine, and latency are logged. Phone numbers and emails in stored text are masked.

---

## 9. Evaluation harness (the portfolio centerpiece)

**Datasets** (download scripts, not committed):
- UCI SMS Spam Collection (spam vs ham SMS).
- A public phishing-email dataset (e.g. from Kaggle / Hugging Face) for email.
- A hand-labeled set of 150–300 recent real-world scam texts (toll, delivery, job, crypto) with `scam_type` labels — the most valuable piece, because public datasets predate today's scams. Remove personal data before saving.

Split: 60% calibration-fit / 40% test for local engines; Jev is evaluated on the same test split.

**Metrics per engine and per question:**
- Accuracy, macro-F1 (`scam_type`), ROC-AUC (`is_scam`).
- **Calibration:** Expected Calibration Error (10 bins), Brier score, reliability diagram.
- **Selective prediction:** accuracy vs coverage curve as the abstain band widens; report accuracy at 80% and 90% coverage.
- p50/p95 latency, and cost per 1,000 checks (Jev input-token pricing; local = $0 + hardware noted).

Output: `eval/results/<date>-<engine>.json` plus PNG charts; the web app's Benchmarks page renders the JSON.

---

## 10. Web client

1. **Check** — big text box, "or upload a screenshot", channel picker (Text / Email / Social DM), optional "who does it claim to be from?".
2. **Result** — a large verdict card (red / amber / green), the probability shown as plain language ("very likely a scam") with the number underneath, red-flag checklist, "What to do now" steps, and a "Was this right?" feedback row.
3. **Learn** — one card per scam type with the tell-tale signs (static content).
4. **Benchmarks** — engine comparison table and charts from §9.
5. **Settings** — engine selection (Jev / local LLM / NLI), Jev API key status.

Accessible by default: large type, high contrast, works at 360px wide, verdict never conveyed by color alone.

---

## 11. Phases and acceptance criteria

### Phase 1 — Decision layer + Jev
Scope: engine interface, question set, preprocessing, Jev engine, verdict logic, `POST /api/check`, eval harness with the SMS dataset, CLI.

Done when:
- Unit tests cover every verdict branch, including the abstain band, using fake engines.
- Jev engine tests use mocked HTTP (respx) for success, 429 retry, timeout, and over-limit truncation.
- `python -m second_look.eval.run --engine jev --dataset sms` produces the full metrics JSON and charts.

### Phase 2 — Open-source engines + calibration
Scope: LLM option-scoring engine, NLI engine, temperature scaling, phishing-email dataset, hand-labeled recent-scam set, engine comparison.

Done when:
- All three engines run through the same harness on the same test split.
- Calibrated local engine's ECE is lower than its uncalibrated ECE on the test split (shown in the report).
- A short `RESULTS.md` states, with numbers, where the open-source engine matches Jev and where it doesn't.

### Phase 3 — App
Scope: web client screens, OCR for screenshots, feedback capture, Benchmarks page, Docker Compose.

Done when:
- `docker compose up` runs the full app; with no Jev key, it falls back to the local engine and says so.
- Playwright E2E: paste a known scam → `LIKELY_SCAM` with at least one red flag; paste a normal message → `PROBABLY_SAFE`; paste an ambiguous message → `UNSURE` with verification steps.
- A screenshot of a scam text yields the same verdict as the pasted text.

---

## 12. Working agreement for Claude Code

- Read this spec fully, write PLAN.md, and wait for approval before coding.
- One phase at a time; tests alongside each feature; `pytest` and `npm test` green before moving on.
- `questions.py` is the single source of truth for questions; `verdict.py` stays free of model calls.
- Never put a real API key in code, tests, or fixtures. `.env.example` only.
- Ask before adding any dependency not in §3.