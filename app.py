"""Personal Assistant: Streamlit entry point.  Run with:  streamlit run app.py"""
import hmac
from pathlib import Path

import streamlit as st

# Resolve files from this file's own folder, so it works wherever the repo is cloned.
HERE = Path(__file__).resolve().parent
LOGO, ICON, ICON_SVG = (str(HERE / "assets" / n) for n in ("logo.svg", "icon.png", "icon.svg"))

st.set_page_config(page_title="Personal Assistant", page_icon=ICON, layout="wide")

from ui.state import _core, boot  # noqa: E402
from ui.theme import inject_css  # noqa: E402

inject_css()
settings, _db, _ret = _core()

# Optional password gate: set APP_PASSWORD in .env or in the cloud app's secrets.
if settings.app_password and not st.session_state.get("authed"):
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.write("")
        st.image(LOGO, width=220)
        with st.form("login"):
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary", width="stretch"):
                if hmac.compare_digest(pw.encode(), settings.app_password.encode()):
                    st.session_state.authed = True
                    st.rerun()
                st.error("That password isn't right.")
    st.stop()

ctx = boot()
st.logo(LOGO, icon_image=ICON_SVG)

nav = st.navigation([
    st.Page("views/assistant.py", title="Assistant", icon=":material/chat_bubble:", default=True),
    st.Page("views/planner.py", title="Planner", icon=":material/event_available:"),
    st.Page("views/documents.py", title="Documents", icon=":material/description:"),
    st.Page("views/memory.py", title="Memory", icon=":material/psychology:"),
    st.Page("views/insights.py", title="Insights", icon=":material/monitoring:"),
    st.Page("views/settings.py", title="Settings", icon=":material/settings:"),
])

with st.sidebar:
    ok = bool(ctx.agent.llm.available)
    label = "AI connected" if ok else "AI not connected"
    st.markdown(f'<span class="pill {"" if ok else "off"}"><span class="dot"></span>{label}</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="side-foot">{ctx.now:%A, %d %B %Y}</div>', unsafe_allow_html=True)

nav.run()
