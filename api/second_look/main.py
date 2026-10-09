"""FastAPI app."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Second Look", version="0.1.0")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
