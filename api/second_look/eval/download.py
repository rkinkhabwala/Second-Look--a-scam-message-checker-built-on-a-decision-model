"""Download eval datasets into eval/data/ (gitignored; never committed).

    python -m second_look.eval.download sms
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import httpx

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"

SMS_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SMS_MEMBER = "SMSSpamCollection"
SMS_SHA256 = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"
SMS_FILE = "sms/SMSSpamCollection.tsv"


def download_sms(data_dir: Path = DATA_DIR) -> Path:
    target = data_dir / SMS_FILE
    if target.exists():
        return target
    print(f"Downloading UCI SMS Spam Collection from {SMS_URL} ...")
    resp = httpx.get(SMS_URL, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        raw = zf.read(SMS_MEMBER)  # read the one file we need by name; never extractall
    digest = hashlib.sha256(raw).hexdigest()
    if SMS_SHA256 and digest != SMS_SHA256:
        raise SystemExit(f"Checksum mismatch for {SMS_MEMBER}: got {digest}, expected {SMS_SHA256}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    print(f"Saved {target} ({len(raw):,} bytes, sha256 {digest})")
    return target


PHISHING_URL = (
    "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/resolve/"
    "{revision}/Phishing_Email.csv"
)
PHISHING_REVISION = "34085a032c123ca237f314a01a67909cdea35e34"
PHISHING_SHA256 = "18ef4fff1acb8986f0ab01e83ede6025420c829656af5a377560fce17e153b97"
PHISHING_FILE = "phishing/Phishing_Email.csv"


def download_phishing(data_dir: Path = DATA_DIR) -> Path:
    """zefang-liu/phishing-email-dataset (LGPL-3.0), a copy of Kaggle's "Phishing Email Detection"."""
    target = data_dir / PHISHING_FILE
    if target.exists():
        return target
    url = PHISHING_URL.format(revision=PHISHING_REVISION or "main")
    print(f"Downloading phishing-email dataset from {url} ...")
    resp = httpx.get(url, follow_redirects=True, timeout=120)
    resp.raise_for_status()
    digest = hashlib.sha256(resp.content).hexdigest()
    if PHISHING_SHA256 and digest != PHISHING_SHA256:
        raise SystemExit(f"Checksum mismatch for {PHISHING_FILE}: got {digest}, expected {PHISHING_SHA256}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(resp.content)
    print(f"Saved {target} ({len(resp.content):,} bytes, sha256 {digest})")
    return target


DATASETS = {"sms": download_sms, "phishing": download_phishing}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    args = parser.parse_args(argv)
    DATASETS[args.dataset]()


if __name__ == "__main__":
    main(sys.argv[1:])
