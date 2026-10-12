# Second Look: project overview

You paste a suspicious text, email or DM. Second Look tells you:
- how likely it is to be a scam,
- what kind of scam it is,
- which warning signs it found,
- what to do next.

When it isn't sure, it says so instead of guessing.

This page explains what we're building, why it's built this way, and where it stands. The full requirements are in [spec.md](spec.md), and the step-by-step build plan and decisions are in [PLAN.md](PLAN.md).

*Last updated 2026-10-12.*

---

## 1. The problem

Scam texts and emails cost people billions a year. Common ones include fake toll bills, "your parcel is held" fees, fake bank alerts, job offers and crypto pitches.

The people hit hardest are older adults, recent immigrants and first-time job seekers, and they're the least likely to spot the pattern.

Today's spam filters are **binary and silent**. They hide a message or they don't, and they never explain why.

**Second Look's answer:**
- an **honest probability** ("87% likely a scam") instead of a yes/no;
- the **specific red flags** found, so the person learns the pattern;
- **"Not sure, here's how to check"** when the evidence is mixed, instead of a confident wrong answer;
- **clear next steps**, always including how to report it (reportfraud.ftc.gov, or forward the text to 7726).

## 2. The core idea: decisions, not writing

Every judgement we need is **multiple choice**:
- Is it a scam?
- Which kind?
- Does it rush you?
- Does it ask for gift cards?
- Does a link go somewhere other than the sender it claims to be?

So we use **decision models**, which pick from fixed options and give a probability for each. We don't use a chatbot that writes free text.

That gives us three properties:

| Property | Why it matters |
|---|---|
| **Nothing is made up** | Every sentence the user sees comes from a template we wrote. The model only chooses options, so it can't invent advice. |
| **We know how sure it is** | Each answer has a probability, so the app can act on confident answers and admit doubt on the rest. |
| **Engines are swappable** | Any engine that answers the same questions can plug in. That lets us compare a paid engine with free local ones. |

## 3. How a check works

```
 Message ──► 1. Preprocess ──► 2. Decision engine ──► 3. Verdict ──► Result card
             (plain code)        (one of three)        (plain code)
```

**1. Preprocess** ([preprocess.py](api/second_look/preprocess.py)). Ordinary code cleans the text and pulls out:
- links, and the real domain behind each one (`usps.com.track-fee.top` is really `track-fee.top`)
- shortened links, and look-alike domains (`paypa1.com`, punycode tricks)
- phone numbers
- money amounts

These facts are **never** guessed by a model. Very long emails are trimmed from the middle, and the result records that it was trimmed.

**2. Decision engine.** It answers **9 fixed questions** ([questions.py](api/second_look/questions.py), the single source of truth):

| Question | Type |
|---|---|
| Is it a scam? | yes/no |
| Which kind of scam (13 kinds, from "delivery fee" to "family emergency")? | multiple choice |
| Does it pressure you to act immediately? | yes/no |
| Which payment method does it ask for (gift card, crypto, wire/Zelle, card on a link, none)? | multiple choice |
| Does it ask for a password, PIN or code? | yes/no |
| Who does it claim to be (bank, government, delivery company…)? | multiple choice |
| Does a link's domain not match the claimed sender? | yes/no |
| Is the offer unrealistically generous? | yes/no |
| Overall risk, from "safe" to "scam" | 5-level scale |

**3. Verdict** ([verdict.py](api/second_look/verdict.py)). Plain, tested rules, with no model involved:
- P(scam) **≥ 0.85** gives **Likely scam**.
- P(scam) **≤ 0.25** gives **Probably safe**, which always adds "this can be wrong, here's how to verify".
- Anything in between gives **Not sure**, which adds "here's how to check".
- **Red flags:** every warning sign the engine is at least 60% sure of, shown as a fixed sentence (e.g. "Asks for payment in gift cards — no real company or agency does this.").
- **Next steps:** chosen by scam type, plus the reporting steps every time.

## 4. The three engines

| Engine | What it is | Runs where | Cost | Speed per check |
|---|---|---|---|---|
| **Jev** | TypeSafe's commercial decision model (`jev-1.13.0`) | TypeSafe's API | ~$0.05 per 1,000 checks | ~0.19 s |
| **LLM option-scoring** | Open model Qwen3-1.7B. We read how likely it is to pick each option letter; no text is generated. | Your computer (GPU) | Free | ~2 s |
| **NLI** | DeBERTa-v3 zero-shot classifier. Tests each option as a sentence ("This message is a toll scam.") | Your computer (CPU) | Free | ~1–2 s for texts, ~6 s for long emails |

The app uses Jev when an API key is set. Otherwise it **falls back to a local engine and says so**. Local engines need a one-time model download (`python -m second_look.engines.weights download llm`).

## 5. How we know it works: the evaluation

We don't trust any engine's confidence without checking it. The evaluation harness ([api/second_look/eval/](api/second_look/eval/)) scores every engine on the **same held-out messages**.

**Datasets**
- **UCI SMS Spam Collection:** 5,160 unique texts.
- **Phishing-email dataset:** a fixed sample of 2,000 emails.
- **Hand-labelled recent scams (coming):** today's toll, delivery, job and crypto scams. This is the most valuable one, because the public sets are 10–20 years old.

Each dataset is split once with a fixed seed:
- **60% for calibration:** tuning an engine's probabilities and thresholds.
- **40% for testing:** final scores, never used for tuning.

**What we measure**

| Metric | Plain meaning |
|---|---|
| Accuracy, ROC-AUC | How often it's right; how well it ranks scams above normal messages |
| **ECE (calibration error)** | Does "90% sure" really mean right 90% of the time? Lower is better. |
| **Accuracy at 80% / 90% coverage** | If the app answers only the messages it's most sure about, how accurate is it? |
| Latency, cost | p50/p95 time per check, and dollars per 1,000 checks |

**Jev results so far** (test splits):

| | SMS (2,064 texts) | Phishing (800 emails) |
|---|---|---|
| Accuracy | 96.9% | 94.8% |
| ROC-AUC | 0.981 | 0.987 |
| ECE | 0.102 | 0.074 |
| Median latency | 190 ms | 184 ms |
| Cost per 1,000 checks | $0.052 | $0.075 |

With the current thresholds, Jev answers **84%** of SMS messages and is **99.4%** accurate on those. It says "Not sure" for the rest.

The local engines' results are being computed now, and will appear in `RESULTS.md` with a side-by-side comparison.

## 6. Key decisions (and why)

| Decision | Why |
|---|---|
| Thresholds tuned from data: "scam" at ≥ 0.85, confidence gate off | The original 0.75 cut-off plus a confidence gate sent 23% of messages to "Not sure". 198 of the 199 messages held back by the gate alone would have been answered correctly. Tuning on the calibration split, then checking on the test split, raised coverage from 77% to 84% at slightly higher accuracy. |
| **Platt scaling** for the local engines' yes/no answers (the spec originally said temperature scaling) | The local model leans towards "yes". Temperature scaling alone made its calibration *worse* (ECE 0.18 → 0.21). Adding one shift per question fixed it (ECE 0.06, accuracy 82% → 96%). Spec §5 records this change. |
| Qwen3-1.7B, not 4B | The 4B model didn't fit alongside everything else in 16 GB of memory (9–51 s per check). After calibration, 1.7B was just as accurate and runs in ~2 s. |
| NLI on CPU | On this Mac, the GPU path was no faster for this model and twice hung mid-run. |
| Tune on the calibration split, report on the test split | Choosing settings on the same data you report on makes results look better than they are. |
| Duplicates removed before splitting | The same message in both halves would leak answers from tuning into testing. |

## 7. Privacy and safety rules

- **Links are never visited.** URLs are analysed as text only.
- **By default only the verdict, engine, latency and a truncation flag are stored.** Message text is kept only if the user ticks "help improve", and then phone numbers and email addresses are masked first.
- **No API keys in code, tests or commits.** Keys live in a local `.env` file that git ignores. The repo has only `.env.example`.
- **Tests never call real APIs.** Any unmocked web request fails the test run.
- **Eval results contain message IDs, never message text.**

## 8. Project status

| Phase | Scope | Status |
|---|---|---|
| **1. Decision layer + Jev** | Engine interface, questions, preprocessing, Jev engine, verdict rules, API, storage, eval harness | ✅ **Done.** All acceptance criteria met. |
| **2. Open-source engines + calibration** | LLM and NLI engines, calibration, phishing and hand-labelled data, engine comparison, `RESULTS.md` | 🟡 **In progress.** Engines, calibration and datasets are built. Full local evaluation is running (~5 h). Still to do: comparison, `RESULTS.md`, and the hand-labelled set. |
| **3. App** | Web app (Check, Result, Learn, Benchmarks, Settings), screenshot OCR, feedback, Docker Compose, end-to-end tests | ⏳ **Not started.** |

**Open items**
- **Hand-labelled set:** 150–300 recent scam messages plus about 100 real legitimate ones, collected by the team with personal details removed. The tooling is ready (see §10).
- **Thresholds for the local engines:** tuned once their runs finish.

## 9. Repository map

```
spec.md              requirements (source of truth)
PLAN.md              build plan, decisions, open questions per phase
PROJECT.md           this overview
api/
  second_look/
    main.py          web API: /api/check, /api/engines, /api/checks/{id}/feedback, /api/health
    questions.py     the 9 questions (single source of truth)
    preprocess.py    text cleaning; link/phone/amount extraction; trimming
    verdict.py       deterministic verdict rules + every user-facing sentence
    calibration.py   Platt / temperature scaling for local engines
    storage.py       SQLite storage with privacy masking
    config.py        thresholds, model versions, settings
    engines/         jev.py, llm_scoring.py, nli.py, registry.py (+ helpers)
    eval/            datasets, metrics, charts, run / sweep / compare CLIs, results/
  tests/             ~190 automated tests (pytest)
```

## 10. Common tasks

All commands run from the `api/` folder. You need [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                                                    # install
uv run pytest                                              # run tests (~10 s)
uv run --env-file ../.env uvicorn second_look.main:app     # start the API
```

**Evaluation**

```bash
uv run python -m second_look.eval.download sms             # fetch a dataset (not committed)
uv run --env-file ../.env python -m second_look.eval.run --engine jev --dataset sms --dry-run   # cost, no calls
uv run python -m second_look.eval.run --engine llm --dataset sms --calibrate                   # local engine
uv run python -m second_look.eval.compare                  # side-by-side table + charts
```

**Hand-labelled data:** add messages to `api/second_look/eval/data/handlabeled/handlabeled.jsonl`, one per line:

```json
{"text": "...", "label": "scam", "scam_type": "toll_or_fine", "channel": "sms", "received": "2026-09", "source": "own phone"}
```

```bash
uv run python -m second_look.eval.handlabeled mask         # mask phone numbers and emails
uv run python -m second_look.eval.handlabeled validate     # check format, flag leftover personal data
```

## 11. Glossary

- **Decision model:** a model that picks among fixed options and gives each a probability, instead of writing text.
- **Calibration:** adjusting probabilities so "X% sure" matches how often it's actually right.
- **ECE (Expected Calibration Error):** the average gap between stated confidence and actual accuracy. 0 is perfect.
- **Coverage:** the share of messages the app gives a firm answer on, instead of "Not sure".
- **Abstain band:** the range of probabilities (0.25–0.85) where the app says "Not sure".
- **Platt scaling:** calibration that rescales *and shifts* a model's yes/no score.
- **Calibration / test split:** the part of a dataset used for tuning vs. the untouched part used for final scores.
