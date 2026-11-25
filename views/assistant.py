"""Assistant: the conversation."""
import streamlit as st

from ui.components import AVATAR_AI, AVATAR_USER, render_plan, render_steps, safety_line
from ui.state import boot, technical
from ui.theme import esc, glance, greeting

ctx = boot()
agent, db, s, now = ctx.agent, ctx.db, ctx.settings, ctx.now
tech = technical()

BRIEF = ("Give me a morning briefing: my pending high-priority tasks, today's and tomorrow's calendar events, "
         "pending reminders, and today's weather in Bhubaneswar. Point out any conflicts or urgent deadlines.")


def suggestions():
    out = []
    if db.query("SELECT 1 FROM tasks WHERE status='pending' LIMIT 1"):
        out.append(("Plan my next three days",
                    "Time-block all my pending tasks into free calendar slots over the next 3 days."))
    out.append(("Brief me for today", BRIEF))
    if db.query("SELECT 1 FROM facts WHERE lower(fact) LIKE '%advis%' LIMIT 1"):
        out.append(("Email my advisor an update",
                    "Look up my advisor's email from memory, check my high-priority tasks, and draft an email update to her."))
    docs = ctx.retriever.list_docs()
    if any("handbook" in d["name"].lower() for d in docs):
        out.append(("What the handbook says about attendance",
                    "What does the university handbook say about attendance and capstone project evaluation?"))
    elif docs:
        out.append(("Summarise my documents", "Search my documents and summarise the key points I should know."))
    return out[:4]


def handle_result(res):
    st.session_state.pending = res["pending"]
    st.session_state.partial = res
    if not res["pending"]:
        st.session_state.chat.append({"role": "assistant", "content": res["reply"], "events": res["events"],
                                      "plan": res["plan"], "stats": res.get("stats")})


def run_turn(text: str):
    st.session_state.chat.append({"role": "user", "content": text})
    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(text)
    with st.chat_message("assistant", avatar=AVATAR_AI):
        with st.spinner("Working on it…"):
            handle_result(agent.send(text))
    st.rerun()


# ------------------------------------------------------------------ header
st.markdown(f'<h1 class="page-title">{esc(greeting(now, s.user_name))}</h1>'
            f'<p class="page-sub">{esc(glance(db, now))}</p>', unsafe_allow_html=True)

for r in db.query("SELECT * FROM reminders WHERE done=0 AND remind_at <= ? ORDER BY remind_at",
                  (now.strftime("%Y-%m-%d %H:%M"),)):
    a, b = st.columns([6, 1], vertical_alignment="center")
    when = r["remind_at"][11:] if r["remind_at"][:10] == now.strftime("%Y-%m-%d") else r["remind_at"]
    a.markdown(f'<div class="reminder">Reminder: {esc(r["text"])} <span style="opacity:.75">({when})</span></div>',
               unsafe_allow_html=True)
    if b.button("Done", key=f"rem_{r['id']}", width="stretch"):
        db.execute("UPDATE reminders SET done=1 WHERE id=?", (r["id"],))
        st.rerun()

if not agent.llm.available:
    with st.container(border=True):
        st.markdown("**Connect an AI provider to start.** Add a Groq or Gemini key, then come back here.")
        st.page_link("views/settings.py", label="Open Settings", icon=":material/settings:")

queued = st.session_state.pop("queued", None)
pending = st.session_state.pending

# ------------------------------------------------------------------ empty state
if not st.session_state.chat and not pending and not queued and agent.llm.available:
    st.markdown('<div class="hello">What would you like to do?</div>'
                '<div class="hello-sub">Ask in your own words. I can manage your tasks and calendar, '
                'search your documents, and draft emails for your approval.</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (label, prompt_text) in enumerate(suggestions()):
        if cols[i % 2].button(label, key=f"sg{i}", width="stretch"):
            st.session_state.queued = prompt_text
            st.rerun()

# ------------------------------------------------------------------ conversation
for m in st.session_state.chat:
    with st.chat_message(m["role"], avatar=AVATAR_USER if m["role"] == "user" else AVATAR_AI):
        if m.get("plan"):
            render_plan(m["plan"])
        st.markdown(m["content"].replace("\n•", "\n\n•") if m["role"] == "assistant" else m["content"])
        render_steps(m.get("events"), m.get("stats"), tech)

if pending:
    partial = st.session_state.get("partial") or {}
    with st.chat_message("assistant", avatar=AVATAR_AI):
        render_plan(partial.get("plan"))
        render_steps(partial.get("events"), None, tech)
        with st.container(border=True, key="approval"):
            st.markdown('<div class="section-title">Needs your approval</div>', unsafe_allow_html=True)
            st.markdown(agent.tools.describe_action(pending["name"], pending["args"]))
            st.markdown(safety_line(pending.get("audit") or {}), unsafe_allow_html=True)
            ok, no, _ = st.columns([1, 1, 3])
            if ok.button("Approve", type="primary", icon=":material/check:", width="stretch"):
                with st.spinner("Carrying it out…"):
                    handle_result(agent.resume(True))
                st.rerun()
            if no.button("Decline", width="stretch"):
                with st.spinner("Updating…"):
                    handle_result(agent.resume(False))
                st.rerun()

# ------------------------------------------------------------------ input (text or voice)
blocked = bool(pending) or not agent.llm.available
hint = ("Approve or decline the action above to continue" if pending
        else "Connect an AI provider in Settings to chat" if not agent.llm.available
        else "Ask anything, or tell me what to do")
entry = st.chat_input(hint, accept_audio=True, disabled=blocked)

text = queued
if entry:
    if isinstance(entry, str):
        text = entry
    else:
        text = (getattr(entry, "text", None) or (entry.get("text") if hasattr(entry, "get") else None) or "").strip()
        audio = getattr(entry, "audio", None) or (entry.get("audio") if hasattr(entry, "get") else None)
        if audio is not None and not text:
            try:
                with st.spinner("Listening…"):
                    text = agent.llm.transcribe(audio.getvalue(), getattr(audio, "name", None) or "voice.wav")
            except Exception as e:
                st.error(f"I couldn't understand that recording: {e}")
                text = None
if text:
    run_turn(text)
