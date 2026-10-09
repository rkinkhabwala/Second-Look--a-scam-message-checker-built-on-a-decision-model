# Second Look

Paste a suspicious text, email or DM and get back how likely it is to be a scam, what kind,
which red flags are present, and what to do next. Decisions come from TypeSafe's **Jev**
decision model (open-source engines arrive in Phase 2). See [spec.md](spec.md) and [PLAN.md](PLAN.md).

## How to run

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
cd api
uv sync                                   # install deps into .venv
cp ../.env.example ../.env                # then put your TypeSafe key in ../.env
uv run --env-file ../.env uvicorn second_look.main:app --reload
```

Health check: `curl localhost:8000/api/health`

### Tests

```bash
cd api
uv run pytest
```

Tests never call the real Jev API; HTTP is mocked with respx and any unmocked request fails.

### Evaluation

```bash
cd api
uv run python -m second_look.eval.download sms                       # UCI SMS Spam Collection -> eval/data (gitignored)
uv run --env-file ../.env python -m second_look.eval.run --engine jev --dataset sms --dry-run   # request count + cost, no calls
uv run --env-file ../.env python -m second_look.eval.run --engine jev --dataset sms --limit 50  # small trial
uv run --env-file ../.env python -m second_look.eval.run --engine jev --dataset sms             # full test split
```

Results (metrics JSON, per-message predictions without text, reliability and coverage PNGs) go to
`api/second_look/eval/results/`. Responses are cached in `eval/cache/` by request hash, so reruns are free.
