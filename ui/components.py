"""Reusable pieces: the plan strip, the step log and the approval card."""
from __future__ import annotations

import json

import streamlit as st

from .theme import esc, step_label

AVATAR_USER = ":material/person:"
AVATAR_AI = ":material/auto_awesome:"


def render_plan(plan) -> None:
    if plan:
        items = "".join(f"<li>{esc(str(step))}</li>" for step in plan)
        st.markdown(f'<div class="plan"><b>My plan</b><ol>{items}</ol></div>', unsafe_allow_html=True)


def render_steps(events, stats=None, technical: bool = False) -> None:
    """A tidy, collapsed log of what the assistant did to produce an answer."""
    if not events:
        return
    n = len(events)
    with st.expander(f"How I got this ({n} step{'s' if n != 1 else ''})"):
        rows = []
        for e in events:
            what = esc(step_label(e["tool"]))
            if technical:
                what += f' <span class="muted">· {esc(e["tool"])}</span>'
            result = esc(str(e["result"]).replace("\n", " ")[:170])
            if technical and e.get("args"):
                result = f'<code>{esc(json.dumps(e["args"], ensure_ascii=False)[:150])}</code><br>{result}'
            ms = f'{e["ms"]:.0f} ms' if e.get("ms") else ""
            rows.append(f'<div class="s"><span class="what">{what}</span><span class="res">{result}</span>'
                        f'<span class="ms">{ms}</span></div>')
        st.markdown('<div class="steps">' + "".join(rows) + "</div>", unsafe_allow_html=True)
        if technical and stats:
            st.caption(f"{stats.get('latency_ms', 0) / 1000:.1f} s of model time · {stats.get('tokens', 0)} tokens")


def safety_line(audit: dict) -> str:
    verdict, score = audit.get("verdict"), audit.get("score")
    note = esc(str(audit.get("critique") or "").strip())
    if verdict == "UNVERIFIED" or score is None:
        return f'<div class="safety"><b>Safety review unavailable.</b> Please read the details carefully. {note}</div>'
    if verdict == "REVISE":
        return f'<div class="safety"><b>Safety review flagged this ({score}/100).</b> {note}</div>'
    return f'<div class="safety"><b>Safety review passed ({score}/100).</b> {note}</div>'
