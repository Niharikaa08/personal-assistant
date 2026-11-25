"""Visual identity and small presentation helpers shared by every page."""
from __future__ import annotations

import html
from datetime import date, datetime, timedelta

import streamlit as st

esc = html.escape

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&display=swap');

:root {
  --ink:#1B2233; --muted:#5E687A; --line:#E2E6ED; --surface:#FFFFFF; --canvas:#F5F6F8;
  --accent:#2F4CC4; --accent-soft:#E8ECFB; --mark:#FFE9A8; --mark-ink:#6B4E00;
  --ok:#1F7A4D; --bad:#B42318;
}
.stApp, .stApp input, .stApp textarea, .stApp button, .stApp select {
  font-family:'Figtree', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
}
.block-container { max-width:1240px; padding-top:2.2rem; padding-bottom:6rem; }

/* ---------- page header ---------- */
.stApp h1.page-title, .page-title { font-size:1.85rem !important; font-weight:700; letter-spacing:-0.02em; margin:0 0 .25rem !important; padding:0 !important; color:var(--ink); line-height:1.2; }
.page-sub   { color:var(--muted); font-size:1rem; margin:0 0 1.5rem; max-width:68ch; line-height:1.5; }
.section-title { font-size:1.05rem; font-weight:650; margin:0 0 .6rem; color:var(--ink); }
.muted { color:var(--muted); font-size:.92rem; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { border-right:1px solid var(--line); }
.pill { display:inline-flex; align-items:center; gap:.5rem; font-size:.85rem; padding:.3rem .75rem;
        border-radius:999px; background:var(--canvas); color:var(--muted); border:1px solid var(--line); }
.pill .dot { width:.5rem; height:.5rem; border-radius:50%; background:var(--ok); }
.pill.off .dot { background:var(--bad); }
.side-foot { margin-top:.5rem; color:var(--muted); font-size:.85rem; line-height:1.6; }

/* ---------- chat ---------- */
[data-testid="stChatMessage"] { background:transparent; padding:.5rem 0; }
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
  background:var(--accent-soft); border-radius:14px; padding:.75rem 1rem; margin-left:12%;
}
.plan { border-left:3px solid var(--accent); padding:.15rem 0 .15rem .8rem; margin:.1rem 0 .8rem; color:var(--muted); font-size:.92rem; }
.plan b { color:var(--ink); font-weight:600; }
.plan ol { margin:.25rem 0 0 1.1rem; padding:0; }
.steps { font-size:.9rem; }
.steps .s { display:flex; gap:.6rem; padding:.28rem 0; border-bottom:1px solid var(--line); align-items:baseline; }
.steps .s:last-child { border-bottom:0; }
.steps .what { font-weight:600; min-width:13rem; color:var(--ink); }
.steps .res { color:var(--muted); flex:1; overflow-wrap:anywhere; }
.steps .ms { color:var(--muted); font-variant-numeric:tabular-nums; white-space:nowrap; }

/* ---------- empty state ---------- */
.hello { font-size:1.15rem; font-weight:600; margin:1.2rem 0 .2rem; }
.hello-sub { color:var(--muted); margin-bottom:1.1rem; }

/* ---------- approval card ---------- */
.st-key-approval { border-left:4px solid #E0A800 !important; }
.safety { font-size:.92rem; color:var(--muted); margin:.2rem 0 .8rem; }
.safety b { color:var(--ink); }

/* ---------- reminder ---------- */
.reminder { background:var(--mark); color:var(--mark-ink); padding:.55rem .85rem; border-radius:10px; font-weight:500; }

/* ---------- agenda ---------- */
.agenda .day { margin-bottom:1.15rem; }
.agenda .day-h { font-weight:650; font-size:.95rem; margin-bottom:.3rem; }
.agenda .day-h small { color:var(--muted); font-weight:500; margin-left:.4rem; }
.agenda .row { display:grid; grid-template-columns:6.6rem 1fr; gap:.6rem; padding:.5rem .7rem;
               border-radius:10px; align-items:center; }
.agenda .row:hover { background:var(--canvas); }
.agenda .time { color:var(--muted); font-variant-numeric:tabular-nums; font-size:.9rem; }
.agenda .note { color:var(--muted); font-size:.85rem; }
.tag { margin-left:.5rem; font-size:.72rem; padding:.12rem .55rem; border-radius:999px; background:var(--accent-soft); color:var(--accent); font-weight:600; }
.tag.late { background:var(--mark); color:var(--mark-ink); }

/* ---------- documents ---------- */
.passage { border-left:3px solid var(--accent-soft); padding:.1rem 0 .1rem .9rem; line-height:1.55; }
.src { font-weight:600; }
.scores { color:var(--muted); font-size:.85rem; font-variant-numeric:tabular-nums; }

/* ---------- tidy native widgets ---------- */
[data-testid="stMetric"] { padding:.15rem 0; }
[data-testid="stMetricLabel"] p { color:var(--muted); font-size:.9rem; }
[data-testid="stMetricValue"] { font-size:1.75rem; font-weight:650; }
[data-testid="stMetricDelta"] { max-width:100%; }
button[kind="primary"] { font-weight:600; }
[data-testid="stExpander"] { border-radius:12px; }
.stTabs [data-baseweb="tab"] { font-weight:600; }
a { text-decoration:none; }
@media (max-width:760px) {
  .block-container { padding-top:1.2rem; }
  .agenda .row { grid-template-columns:5.6rem 1fr; }
  .stApp h1.page-title { font-size:1.5rem !important; }
  [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) { margin-left:0; }
  .steps .what { min-width:8rem; }
}
</style>
"""

TOOL_LABELS = {
    "make_plan": "Planned the steps",
    "add_task": "Added a task",
    "list_tasks": "Checked your tasks",
    "complete_task": "Completed a task",
    "delete_task": "Deleted a task",
    "add_event": "Added a calendar event",
    "list_events": "Checked your calendar",
    "delete_event": "Removed a calendar event",
    "find_free_slots": "Looked for free time",
    "auto_schedule_tasks": "Scheduled focus time",
    "set_reminder": "Set a reminder",
    "list_reminders": "Checked your reminders",
    "remember_fact": "Saved to memory",
    "forget_fact": "Removed from memory",
    "search_documents": "Searched your documents",
    "web_search": "Searched the web",
    "get_weather": "Checked the weather",
    "send_email": "Prepared an email",
    "🛡️ critic_audit": "Safety review",
}


def clean_title(title: str) -> str:
    """Hide scheduler bookkeeping such as "[Focus] ... (high)" from people."""
    import re
    return re.sub(r"\s*\((high|medium|low)\)$", "", title.replace("[Focus] ", ""))


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, sub: str = "") -> None:
    st.markdown(f'<h1 class="page-title">{esc(title)}</h1><p class="page-sub">{esc(sub)}</p>', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{esc(title)}</div>', unsafe_allow_html=True)


def step_label(tool: str) -> str:
    return TOOL_LABELS.get(tool, tool.replace("_", " ").capitalize())


def greeting(now: datetime, name: str) -> str:
    part = "morning" if now.hour < 12 else "afternoon" if now.hour < 17 else "evening"
    return f"Good {part}, {name}"


def rel_day(d: date, today: date) -> str:
    delta = (d - today).days
    return {0: "Today", 1: "Tomorrow", -1: "Yesterday"}.get(delta, d.strftime("%A"))


def day_label(d: date, today: date) -> str:
    return f"{d.strftime('%a %d %b')}"


def glance(db, now: datetime) -> str:
    """One plain sentence about what is coming up."""
    soon = (now.replace(microsecond=0)).strftime("%Y-%m-%d %H:%M")
    open_n = db.query("SELECT COUNT(*) AS n FROM tasks WHERE status='pending'")[0]["n"]
    due_n = db.query("SELECT COUNT(*) AS n FROM tasks WHERE status='pending' AND due IS NOT NULL AND due <= ?",
                     (_plus_days(now, 2),))[0]["n"]
    nxt = db.query("SELECT title, start FROM events WHERE start >= ? ORDER BY start LIMIT 1", (soon,))
    parts = []
    if open_n:
        parts.append(f"You have {open_n} open task{'s' if open_n != 1 else ''}"
                     + (f", {due_n} due within two days." if due_n else "."))
    else:
        parts.append("Your task list is clear.")
    if nxt:
        start = datetime.strptime(nxt[0]["start"], "%Y-%m-%d %H:%M")
        parts.append(f"Next on your calendar: {clean_title(nxt[0]['title'])}, {rel_day(start.date(), now.date()).lower()} at {start:%H:%M}.")
    return " ".join(parts)


def _plus_days(now: datetime, days: int) -> str:
    return (now + timedelta(days=days)).strftime("%Y-%m-%d 23:59")
