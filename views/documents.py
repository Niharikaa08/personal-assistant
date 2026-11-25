"""Documents: the personal knowledge library."""
import streamlit as st

from assistant.rag import SUPPORTED
from ui.state import boot, technical
from ui.theme import esc, page_header, section

ctx = boot()
retriever = ctx.retriever
tech = technical()

page_header("Documents", "Add your files and the assistant can answer questions from them, with sources.")

left, right = st.columns([2, 3], gap="large")
with left:
    section("Add documents")
    files = st.file_uploader("Choose files", type=[x[1:] for x in sorted(SUPPORTED)], accept_multiple_files=True,
                             label_visibility="collapsed", help="PDF, Word, text or Markdown")
    if files and st.button("Add to library", type="primary", width="stretch"):
        for f in files:
            try:
                with st.spinner(f"Reading {f.name}…"):
                    n = retriever.ingest(f.name, f.getvalue())
                st.toast(f"Added {f.name} ({n} passages)")
            except Exception as e:
                st.error(f"{f.name}: {e}")
        st.rerun()

with right:
    section("Library")
    docs = retriever.list_docs()
    if not docs:
        st.info("Your library is empty. Add a PDF, Word or text file to get started.")
    for d in docs:
        with st.container(border=True):
            a, b = st.columns([5, 1], vertical_alignment="center")
            a.markdown(f"**{d['name']}**  \n<span class='muted'>{d['chunks']} passage{'s' if d['chunks'] != 1 else ''}</span>", unsafe_allow_html=True)
            if b.button("Remove", key=f"rm_{d['name']}", width="stretch"):
                retriever.delete_doc(d["name"])
                st.rerun()

st.divider()
section("Search your documents")
q = st.text_input("Search", placeholder="For example: what is the minimum attendance?", label_visibility="collapsed")
if q:
    hits = retriever.search(q, k=4)
    if not hits:
        st.warning("Nothing in your documents matches that. Try different words.")
    for h in hits:
        with st.container(border=True):
            st.markdown(f"<span class='src'>{esc(h['doc'])}</span> <span class='muted'>passage {h['idx'] + 1}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='passage'>{esc(h['text'])}</div>", unsafe_allow_html=True)
            if tech:
                st.markdown(f"<span class='scores'>combined {h['score']} · semantic {h['dense_score']} · "
                            f"keyword {h['sparse_score']}</span>", unsafe_allow_html=True)
