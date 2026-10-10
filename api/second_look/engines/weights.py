"""Download local model weights and check whether they're cached.

    python -m second_look.engines.weights download llm
    python -m second_look.engines.weights download nli

Weights go to the standard Hugging Face cache, never into the repo. Readiness
checks only look at the cache; they never start a download.
"""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import snapshot_download, try_to_load_from_cache

from second_look.config import Settings, get_settings

# Only what loading needs: no ONNX exports, no pickled training args.
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "LICENSE"]


def model_ref(engine: str, settings: Settings) -> tuple[str, str]:
    if engine == "llm":
        return settings.llm_model, settings.llm_revision
    if engine == "nli":
        return settings.nli_model, settings.nli_revision
    raise KeyError(engine)


def is_cached(repo_id: str, revision: str, cache_dir: str | None = None) -> bool:
    """True if this exact revision's config and weights index (or single weights file) are cached."""
    def cached(name: str) -> bool:
        return isinstance(try_to_load_from_cache(repo_id, name, revision=revision, cache_dir=cache_dir), str)

    return cached("config.json") and (cached("model.safetensors") or cached("model.safetensors.index.json"))


def download(repo_id: str, revision: str, cache_dir: str | None = None) -> str:
    return snapshot_download(repo_id, revision=revision, allow_patterns=ALLOW_PATTERNS, cache_dir=cache_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m second_look.engines.weights", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["download", "status"])
    parser.add_argument("engine", choices=["llm", "nli"])
    args = parser.parse_args(argv)
    repo, rev = model_ref(args.engine, get_settings())
    if args.action == "status":
        print(f"{repo}@{rev[:8]}: {'cached' if is_cached(repo, rev) else 'not downloaded'}")
    else:
        print(f"Downloading {repo}@{rev[:8]} ...")
        print(download(repo, rev))


if __name__ == "__main__":
    main(sys.argv[1:])
