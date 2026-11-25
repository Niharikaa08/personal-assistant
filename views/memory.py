"""Memory: facts the assistant keeps, and the emails it prepared."""
import streamlit as st

from ui.state import boot
from ui.theme import page_header, section

ctx = boot()
db, agent = ctx.db, ctx.agent

page_header("Memory", "Facts the assistant keeps about you. It uses them in every conversation.")

with st.form("add_fact", clear_on_submit=True, border=False):
    a, b = st.columns([5, 1], vertical_alignment="bottom")
    fact = a.text_input("Remember something", placeholder="For example: I prefer studying before noon")
    if b.form_submit_button("Remember", type="primary", width="stretch") and fact.strip():
        st.toast(agent.tools.remember_fact(fact.strip()))
        st.rerun()

facts = db.query("SELECT id, fact FROM facts ORDER BY id DESC")
if not facts:
    st.info("Nothing remembered yet. Tell the assistant about yourself, or add a fact above.")
for r in facts:
    with st.container(border=True):
        a, b = st.columns([6, 1], vertical_alignment="center")
        a.write(r["fact"])
        if b.button("Forget", key=f"fg_{r['id']}", width="stretch"):
            db.execute("DELETE FROM facts WHERE id=?", (r["id"],))
            st.rerun()

st.write("")
section("Emails")
mail = db.query("SELECT subject, to_addr, status, created_at FROM outbox ORDER BY id DESC LIMIT 20")
if not mail:
    st.caption("No emails yet. The assistant always asks before sending one.")
else:
    st.dataframe(
        [{"Subject": r["subject"], "To": r["to_addr"],
          "Status": "Sent" if r["status"] == "sent" else "Saved, not sent", "When": r["created_at"]} for r in mail],
        hide_index=True, width="stretch")
