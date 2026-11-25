"""Settings: AI connection, interface and data."""
import streamlit as st

from assistant.seed import seed_demo
from ui.state import boot, reload_settings, reset_conversation, technical
from ui.theme import page_header, section

ctx = boot()
agent, db, s = ctx.agent, ctx.db, ctx.settings

page_header("Settings", "Connect your AI provider and manage your data.")

# ------------------------------------------------------------------ AI connection
section("AI connection")
st.caption("The assistant tries providers in this order and switches automatically if one fails.")
if not agent.llm.available:
    st.warning("No provider is configured. Add GROQ_API_KEY or GEMINI_API_KEY to your .env file or app secrets.")
for p in agent.llm.providers:
    st.markdown(f"**{p.name.capitalize()}** · key found" + (f" · model pinned to `{p.pinned}`" if p.pinned else " · model chosen automatically"))
for name, why in agent.llm.skipped.items():
    st.markdown(f"**{name.capitalize()}** · not configured. <span class='muted'>{why}</span>", unsafe_allow_html=True)

c1, c2, _ = st.columns([1, 1, 2])
run = c1.button("Test connection", type="primary", icon=":material/network_check:", width="stretch",
                disabled=not agent.llm.available)
if c2.button("Reload settings", icon=":material/refresh:", width="stretch",
             help="Re-read your .env file or secrets after editing them"):
    reload_settings()
    st.rerun()

if run:
    with st.spinner("Checking each provider…"):
        st.session_state.diag = agent.llm.diagnose()
for row in [r for r in st.session_state.get("diag", []) if r["configured"]]:
    with st.container(border=True):
        st.markdown(f"**{row['provider'].capitalize()}**" + (f" · using `{row['model']}`" if row["model"] else ""))
        for c in row["checks"]:
            icon, color = (":material/check_circle:", "green") if c["ok"] else (":material/cancel:", "red")
            st.markdown(f":{color}[{icon}] **{c['check']}** · {c['detail']}")
        if row["hint"]:
            st.info(row["hint"])

st.divider()
section("Interface")
st.session_state["_tech"] = st.toggle("Show technical details", value=technical(),
                                      help="Adds tool names, arguments and retrieval scores to answers and search results.")

st.divider()
section("Your data")
a, b, _ = st.columns([1, 1, 2])
if a.button("Clear conversation", width="stretch"):
    reset_conversation()
    st.toast("Conversation cleared")
    st.rerun()
if b.button("Load sample data", width="stretch", help="Replaces your data with a small demo set"):
    st.session_state["confirm_seed"] = True
if st.session_state.get("confirm_seed"):
    st.warning("This replaces all tasks, events, memory and documents with sample data.")
    y, n, _ = st.columns([1, 1, 3])
    if y.button("Replace my data", type="primary", width="stretch"):
        seed_demo(db, ctx.retriever, s.tz)
        reset_conversation()
        st.session_state.pop("confirm_seed")
        st.rerun()
    if n.button("Cancel", width="stretch"):
        st.session_state.pop("confirm_seed")
        st.rerun()

with st.expander("Erase everything"):
    st.write("Deletes all tasks, events, reminders, memory, documents and logs. This cannot be undone.")
    if st.checkbox("I understand") and st.button("Erase all data", type="primary"):
        db.reset()
        reset_conversation()
        st.rerun()

if s.app_password:
    st.divider()
    if st.button("Sign out"):
        st.session_state.pop("authed", None)
        st.rerun()
