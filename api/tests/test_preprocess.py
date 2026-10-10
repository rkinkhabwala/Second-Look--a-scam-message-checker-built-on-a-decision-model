import json

import pytest

from second_look.preprocess import (
    build_state,
    estimate_tokens,
    extract_amounts,
    extract_phone_numbers,
    extract_urls,
    normalize,
    truncate_middle,
)


def domains(text):
    return [u["registered_domain"] for u in extract_urls(text)]


# --------------------------------------------------------------- normalization


def test_normalize_strips_zero_width_and_collapses_space():
    assert normalize("Pay​ now   \t please\r\n\r\n\r\nok") == "Pay now please\n\nok"


def test_normalize_nfkc_fullwidth():
    assert normalize("ＵＳＰＳ") == "USPS"


def test_normalize_refangs_urls():
    assert normalize("hxxps://evil[.]com/pay") == "https://evil.com/pay"


# ------------------------------------------------------------------------ URLs


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Track: bit.ly/3xYz9", ["bit.ly"]),
        ("see https://t.co/AbC123?x=1.", ["t.co"]),
        ("tinyurl.com/pkg-fee now", ["tinyurl.com"]),
    ],
)
def test_shortened_urls(text, expected):
    urls = extract_urls(text)
    assert [u["registered_domain"] for u in urls] == expected
    assert all(u["shortener"] for u in urls)


def test_shortened_url_keeps_path_and_drops_trailing_punct():
    assert extract_urls("Go to https://t.co/AbC123?x=1.")[0]["url"] == "https://t.co/AbC123?x=1"


@pytest.mark.parametrize(
    "text, host, registered",
    [
        ("Pay at paypa1.com/login", "paypa1.com", "paypa1.com"),
        ("https://usps.com.track-fee.top/pay", "usps.com.track-fee.top", "track-fee.top"),
        ("amaz0n-support.co.uk/verify", "amaz0n-support.co.uk", "amaz0n-support.co.uk"),
        ("www.chase.com-secure.info", "www.chase.com-secure.info", "com-secure.info"),
        ("http://sub.paypal.com.evil.co.uk/x", "sub.paypal.com.evil.co.uk", "evil.co.uk"),
    ],
)
def test_look_alike_domains_resolve_to_real_registered_domain(text, host, registered):
    (u,) = extract_urls(text)
    assert u["host"] == host
    assert u["registered_domain"] == registered
    assert u["shortener"] is False


def test_punycode_domain_is_decoded_alongside():
    (u,) = extract_urls("Login: https://xn--pypal-4ve.com/secure")
    assert u["registered_domain"] == "xn--pypal-4ve.com"
    assert u["unicode_host"] == "pаypal.com"  # Cyrillic 'а'
    assert u["unicode_host"] != "paypal.com"


def test_not_urls():
    assert domains("e.g. it costs 3.99 today. Pay now.Click here") == []


def test_email_address_is_not_a_url():
    assert domains("write to help@usps.com") == []


def test_url_in_parentheses():
    assert extract_urls("(visit https://irs.gov/pay)")[0]["url"] == "https://irs.gov/pay"


def test_duplicate_urls_listed_once():
    assert domains("bit.ly/a then bit.ly/a again") == ["bit.ly"]


# ---------------------------------------------------------------------- phones


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Call (555) 123-4567 now", ["+15551234567"]),
        ("Call +1 555.123.4567", ["+15551234567"]),
        ("ring +44 20 7946 0958", ["+442079460958"]),
        ("call 1-800-555-0199", ["+18005550199"]),
        ("call 5551234567", ["+15551234567"]),
        ("Text STOP to 88022", ["88022"]),
        ("forward it to 7726", ["7726"]),
    ],
)
def test_phone_formats(text, expected):
    assert extract_phone_numbers(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Order #4829104829 shipped",
        "Tracking number: 9400111899223344",
        "Due 2024-10-09 at 10:30",
        "In 2025 we saw 12345 cases",  # numbers without a texting verb aren't short codes
    ],
)
def test_not_phones(text):
    assert extract_phone_numbers(text) == []


# --------------------------------------------------------------------- amounts


@pytest.mark.parametrize(
    "text, value, currency",
    [
        ("fee of $3.99 due", 3.99, "USD"),
        ("owe $1,250 now", 1250.0, "USD"),
        ("send USD 50 today", 50.0, "USD"),
        ("just 50 dollars", 50.0, "USD"),
        ("win $1.2k", 1200.0, "USD"),
        ("pay £20", 20.0, "GBP"),
        ("€ 9.50 toll", 9.5, "EUR"),
        ("earn 2 million dollars", 2_000_000.0, "USD"),
        ("$5 m prize", 5_000_000.0, "USD"),
    ],
)
def test_amount_formats(text, value, currency):
    (a,) = extract_amounts(text)
    assert a["value"] == pytest.approx(value)
    assert a["currency"] == currency


def test_amount_not_confused_with_more():
    assert extract_amounts("$20 more") == [{"text": "$20", "value": 20.0, "currency": "USD"}]


def test_multiple_amounts_in_order():
    assert [a["value"] for a in extract_amounts("Pay $3.99 or 10 dollars")] == [3.99, 10.0]


def test_plain_numbers_are_not_amounts():
    assert extract_amounts("Order 1234 has 3 items") == []


# --------------------------------------------------------------- state & truncation


def test_build_state_shape():
    text = "USPS: your parcel is held. Pay $1.99 at https://usps.com.track-fee.top or call (555) 123-4567"
    state, meta = build_state(text, channel="sms", claimed_sender="  USPS ")
    assert set(state) == {"message", "channel", "claimed_sender", "urls", "phone_numbers", "amounts"}
    assert state["channel"] == "sms"
    assert state["claimed_sender"] == "USPS"
    assert [u["registered_domain"] for u in state["urls"]] == ["track-fee.top"]
    assert state["phone_numbers"] == ["+15551234567"]
    assert [a["value"] for a in state["amounts"]] == [1.99]
    assert meta.truncated is False


def test_digits_inside_urls_are_not_phones_or_amounts():
    state, _ = build_state("https://pay.example.com/5551234567/$20")
    assert state["phone_numbers"] == []
    assert state["amounts"] == []


def test_empty_claimed_sender_is_none():
    assert build_state("hi", claimed_sender="  ")[0]["claimed_sender"] is None


def test_truncate_middle_noop_when_under_limit():
    assert truncate_middle("short", 100) == ("short", False)


def test_truncate_middle_keeps_head_and_tail():
    text = "HEAD " + "filler " * 10_000 + " TAIL"
    out, truncated = truncate_middle(text, 500)
    assert truncated
    assert out.startswith("HEAD ") and out.endswith(" TAIL")
    assert "characters removed" in out
    assert estimate_tokens(out) <= 500


def test_build_state_truncates_and_records_it_but_keeps_late_links():
    text = "Dear customer, " + "lorem ipsum " * 20_000 + " middle link https://evil.top/x " + "dolor " * 20_000
    state, meta = build_state(text, max_state_tokens=2_000)
    assert meta.truncated
    assert meta.original_chars == len(text)
    assert estimate_tokens(state) <= 2_000
    assert "https://evil.top/x" not in state["message"]
    assert [u["registered_domain"] for u in state["urls"]] == ["evil.top"]


def test_build_state_not_truncated_when_under_budget():
    state, meta = build_state("hello there", max_state_tokens=2_000)
    assert state["message"] == "hello there" and not meta.truncated


def test_token_estimate_is_conservative_vs_measured_jev_tokenization():
    from second_look.preprocess import BYTES_PER_TOKEN, MEASURED_JEV_BYTES_PER_TOKEN

    # Fewer assumed bytes per token -> more estimated tokens -> we never under-count Jev's limit.
    assert BYTES_PER_TOKEN < 2.68 <= MEASURED_JEV_BYTES_PER_TOKEN
    payload = {"state": {"message": "Your package is held. Pay $1.99 at usps-redeliver.top"}}
    measured = len(json.dumps(payload, ensure_ascii=False).encode()) / MEASURED_JEV_BYTES_PER_TOKEN
    assert estimate_tokens(payload) >= measured
