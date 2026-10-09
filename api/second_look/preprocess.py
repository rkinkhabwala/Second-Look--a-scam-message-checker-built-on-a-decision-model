"""Build the state object every engine sees (spec §6).

Deterministic features (URLs, domains, phone numbers, amounts) are extracted
here with plain code and never asked of the model. Links are analysed as text
only; nothing here fetches a URL.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

import tldextract

Channel = Literal["sms", "email", "dm"]

MAX_ITEMS_PER_LIST = 50

# Offline: use the public-suffix snapshot bundled with tldextract, never the network.
_tld = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=())

SHORTENERS = frozenset({
    "bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly",
    "cutt.ly", "shorturl.at", "rb.gy", "t.ly", "tiny.cc", "s.id", "v.gd", "lnkd.in",
})

# ---------------------------------------------------------------- normalization

_ZERO_WIDTH = re.compile("[​-‏⁠﻿­]")
_DEFANG = [
    (re.compile(r"\bhxxp(s?)://", re.I), r"http\1://"),
    (re.compile(r"\[\.\]|\(\.\)|\[dot\]", re.I), "."),
]
_SPACES = re.compile(r"[ \t\f\v ]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for pattern, repl in _DEFANG:
        text = pattern.sub(repl, text)
    text = _SPACES.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


# ------------------------------------------------------------------------ URLs

_URL_CANDIDATE = re.compile(
    r"(?<![@\w.-])"  # not part of an email address or a longer token
    r"(?:(?P<scheme>https?://)|(?P<www>www\.))?"
    r"(?P<host>(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9-]{2,63})"
    r"(?P<rest>(?::\d{1,5})?(?:[/?#][^\s<>\"']*)?)",
    re.I,
)
_TRAILING_PUNCT = ".,;:!?)]}'\"…"


def _strip_trailing(url: str) -> str:
    while url and url[-1] in _TRAILING_PUNCT:
        # keep a closing paren that has a matching opener inside the URL
        if url[-1] == ")" and url.count("(") >= url.count(")"):
            break
        url = url[:-1]
    return url


def _decode_idna(host: str) -> str | None:
    if "xn--" not in host:
        return None
    try:
        return host.encode("ascii").decode("idna")
    except UnicodeError:
        return None


def extract_urls(text: str) -> list[dict]:
    """Return one entry per distinct URL: the URL, its host and its registered domain."""
    return [entry for _, entry in _find_urls(text) if entry is not None]


def _find_urls(text: str) -> list[tuple[tuple[int, int], dict | None]]:
    found: list[tuple[tuple[int, int], dict | None]] = []
    seen: set[str] = set()
    for m in _URL_CANDIDATE.finditer(text):
        host = (m.group("www") or "") + m.group("host")
        bare = not (m.group("scheme") or m.group("www"))
        tld_label = host.rsplit(".", 1)[-1]
        # Bare "word.Word" is usually a missing space after a sentence ("now.Click"), not a link.
        if bare and not tld_label.islower():
            continue
        parts = _tld(host.lower())
        registered = parts.top_domain_under_public_suffix
        if not registered:  # "e.g", "3.99", "file.txt"-style tokens with no public suffix
            continue
        url = _strip_trailing(m.group(0))
        span = (m.start(), m.start() + len(url))
        if url.lower() in seen:
            found.append((span, None))  # still masked, not listed twice
            continue
        seen.add(url.lower())
        entry = {
            "url": url,
            "host": host.lower(),
            "registered_domain": registered,
            "shortener": registered in SHORTENERS,
        }
        unicode_host = _decode_idna(host.lower())
        if unicode_host:
            entry["unicode_host"] = unicode_host
        found.append((span, entry))
    return found


def _mask(text: str, spans: list[tuple[int, int]]) -> str:
    """Blank out spans (e.g. URLs) so their digits aren't read as phones or amounts."""
    chars = list(text)
    for start, end in spans:
        for i in range(start, end):
            if chars[i] != "\n":
                chars[i] = " "
    return "".join(chars)


# ---------------------------------------------------------------------- phones

_PHONE_INTL = re.compile(r"(?<![\w+])\+\d{1,3}(?:[\s.-]?\(?\d{1,4}\)?){2,5}(?!\w)")
_PHONE_NANP = re.compile(
    r"(?<![\w#+])(?:1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"
)
# 4–6 digit SMS short codes, only when a texting verb points at them ("Text STOP to 88022").
_SHORT_CODE = re.compile(
    r"\b(?:text|txt|reply|send|sms|forward|msg)\b[^\d\n]{0,25}?(?<![\w#])(\d{4,6})(?!\w)", re.I
)
_ID_CONTEXT = re.compile(r"(?:order|ref|reference|tracking|invoice|case|id|no)\W{0,3}$", re.I)


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def extract_phone_numbers(text: str) -> list[str]:
    """Phone numbers normalized to +<digits> (E.164-like) or bare short codes."""
    out: list[str] = []
    taken: list[tuple[int, int]] = []

    def add(value: str, span: tuple[int, int]) -> None:
        if any(span[0] < e and s < span[1] for s, e in taken):
            return
        taken.append(span)
        if value not in out:
            out.append(value)

    for m in _PHONE_INTL.finditer(text):
        d = _digits(m.group(0))
        if 8 <= len(d) <= 15:
            add("+" + d, m.span())
    for m in _PHONE_NANP.finditer(text):
        if _ID_CONTEXT.search(text[max(0, m.start() - 15) : m.start()]):
            continue
        d = _digits(m.group(0))
        d = d[1:] if len(d) == 11 else d
        add("+1" + d, m.span())
    for m in _SHORT_CODE.finditer(text):
        add(m.group(1), m.span(1))
    return out


# --------------------------------------------------------------------- amounts

_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}
_CODES = {"usd": "USD", "eur": "EUR", "gbp": "GBP", "cad": "CAD", "aud": "AUD"}
_WORDS = {"dollar": "USD", "dollars": "USD", "bucks": "USD", "euro": "EUR", "euros": "EUR",
          "pound": "GBP", "pounds": "GBP", "usd": "USD", "eur": "EUR", "gbp": "GBP"}
_NUM = r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_MULT = r"(?:\s?(?P<mult>k|m|bn|thousand|million|billion)\b)?"
_MULTIPLIERS = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}

_AMOUNT_PATTERNS = [
    # $3.99, $1,250, £20, $1.2k
    re.compile(rf"(?P<sym>[$€£])\s?{_NUM}{_MULT}(?!\w)", re.I),
    # USD 50, CAD$20
    re.compile(rf"(?<![A-Za-z])(?P<code>USD|EUR|GBP|CAD|AUD)\s?\$?{_NUM}{_MULT}(?!\w)", re.I),
    # 50 dollars, 2 million dollars, 20 bucks
    re.compile(
        rf"(?<![\w.,$€£]){_NUM}{_MULT}\s?(?P<word>dollars?|bucks|euros?|pounds?|usd|eur|gbp)\b",
        re.I,
    ),
]


def extract_amounts(text: str) -> list[dict]:
    """Money amounts as {"text", "value", "currency"}, in the order they appear."""
    hits: list[tuple[int, int, dict]] = []
    for pattern in _AMOUNT_PATTERNS:
        for m in pattern.finditer(text):
            if any(m.start() < e and s < m.end() for s, e, _ in hits):
                continue
            g = m.groupdict()
            value = float(g["num"].replace(",", ""))
            value *= _MULTIPLIERS.get((g["mult"] or "").lower(), 1.0)
            if g.get("sym"):
                currency = _SYMBOLS[g["sym"]]
            elif g.get("code"):
                currency = _CODES[g["code"].lower()]
            else:
                currency = _WORDS[g["word"].lower()]
            hits.append((m.start(), m.end(), {"text": m.group(0), "value": value, "currency": currency}))
    return [h[2] for h in sorted(hits, key=lambda h: h[0])]


# ------------------------------------------------------------------ truncation


def estimate_tokens(obj: str | dict | list) -> int:
    """Conservative token estimate (≈3 UTF-8 bytes per token); Jev publishes no tokenizer."""
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return math.ceil(len(s.encode("utf-8")) / 3)


def truncate_middle(text: str, max_tokens: int) -> tuple[str, bool]:
    """Cut from the middle so the opening hook and the closing call-to-action both survive."""
    if estimate_tokens(text) <= max_tokens:
        return text, False
    marker_budget = 16
    keep_bytes = max(0, (max_tokens - marker_budget) * 3)
    encoded = text.encode("utf-8")
    head = encoded[: keep_bytes // 2].decode("utf-8", errors="ignore")
    tail = encoded[len(encoded) - keep_bytes // 2 :].decode("utf-8", errors="ignore")
    removed = len(text) - len(head) - len(tail)
    return f"{head}\n[… {removed} characters removed …]\n{tail}", True


def state_token_budget(
    questions_json: dict, max_state_tokens: int = 32_000, max_request_tokens: int = 64_000
) -> int:
    """Largest state Jev accepts with these questions.

    Jev limits: 64K tokens per request; 32K for state + the longest question.
    """
    longest_q = max((estimate_tokens(q) for q in questions_json.values()), default=0)
    margin = 256  # request envelope, model id, JSON punctuation
    return min(
        max_state_tokens - longest_q, max_request_tokens - estimate_tokens(questions_json)
    ) - margin


# ----------------------------------------------------------------------- state


@dataclass(frozen=True)
class PreprocessMeta:
    truncated: bool
    original_chars: int
    estimated_tokens: int


def build_state(
    text: str,
    channel: Channel = "sms",
    claimed_sender: str | None = None,
    *,
    max_state_tokens: int | None = None,
) -> tuple[dict, PreprocessMeta]:
    """Build the JSON state sent to every engine.

    Features are extracted from the full normalized text; only the `message`
    field is middle-truncated, so a link buried in a long email is still listed.
    `max_state_tokens` is the budget for the whole state object.
    """
    message = normalize(text)
    found = _find_urls(message)
    urls = [entry for _, entry in found if entry is not None]
    without_urls = _mask(message, [span for span, _ in found])
    state = {
        "message": message,
        "channel": channel,
        "claimed_sender": (claimed_sender or "").strip() or None,
        "urls": urls[:MAX_ITEMS_PER_LIST],
        "phone_numbers": extract_phone_numbers(without_urls)[:MAX_ITEMS_PER_LIST],
        "amounts": extract_amounts(without_urls)[:MAX_ITEMS_PER_LIST],
    }
    truncated = False
    if max_state_tokens is not None:
        overhead = estimate_tokens({**state, "message": ""})
        state["message"], truncated = truncate_middle(message, max_state_tokens - overhead)
    return state, PreprocessMeta(
        truncated=truncated, original_chars=len(text), estimated_tokens=estimate_tokens(state)
    )
