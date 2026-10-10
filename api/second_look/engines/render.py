"""Render the state object (spec §6) as plain text for local models.

Jev reads the JSON state directly; local models read this fixed text layout.
"""

from __future__ import annotations

CHANNEL_NAMES = {"sms": "text message", "email": "email", "dm": "social media direct message"}


def render_state(state: str | dict) -> str:
    if isinstance(state, str):
        return f"Message:\n{state}"
    lines = [f"Message ({CHANNEL_NAMES.get(state.get('channel'), 'message')}):", state.get("message", "")]
    lines.append("")
    lines.append(f"Claimed sender: {state.get('claimed_sender') or 'not stated'}")
    urls = state.get("urls") or []
    if urls:
        lines.append("Links in the message:")
        for u in urls:
            notes = [f"registered domain {u['registered_domain']}"]
            if u.get("shortener"):
                notes.append("link shortener")
            if u.get("unicode_host"):
                notes.append(f"displays as {u['unicode_host']}")
            lines.append(f"- {u['url']} ({'; '.join(notes)})")
    else:
        lines.append("Links in the message: none")
    lines.append(f"Phone numbers: {', '.join(state.get('phone_numbers') or []) or 'none'}")
    amounts = state.get("amounts") or []
    lines.append(f"Money amounts: {', '.join(a['text'] for a in amounts) or 'none'}")
    return "\n".join(lines)
