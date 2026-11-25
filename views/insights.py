"""Insights: knowledge graph, performance and (added later) reliability."""
import pandas as pd
import streamlit as st

from ui.state import boot
from ui.theme import page_header

ctx = boot()
db = ctx.db

page_header("Insights", "See what the assistant knows, how it is running, and how reliable it is.")
VIEWS = ["Knowledge graph", "Performance", "Reliability lab"]
view = st.segmented_control("View", VIEWS, default=VIEWS[0], label_visibility="collapsed") or VIEWS[0]

if view == "Knowledge graph":
    st.caption("People, tasks, documents and events the assistant has connected. It grows as you use it.")
    edges = db.query("SELECT source, relation, target, origin FROM kg_edges ORDER BY id DESC LIMIT 60")
    if not edges:
        st.info("Nothing connected yet. Add a document or tell the assistant a fact.")
    else:
        q = lambda x: x.replace('"', "'")
        dot = ["digraph G {", "rankdir=LR;", 'bgcolor="transparent";',
               'node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", fontcolor="#1B2233", color="#C9D1E3", fontname="Sans"];',
               'edge [color="#8A94A8", fontcolor="#5E687A", fontsize=10, fontname="Sans"];']
        dot += [f'"{q(e["source"])}" -> "{q(e["target"])}" [label=" {q(e["relation"].replace("_", " ").lower())} "];' for e in edges]
        dot.append("}")
        st.graphviz_chart("\n".join(dot), width="stretch")
        with st.expander("Show as a table"):
            st.dataframe(pd.DataFrame(edges).rename(columns=str.capitalize), hide_index=True, width="stretch")

elif view == "Reliability lab":
    from ui.reliability import render
    render(ctx)

else:
    metrics = db.query("SELECT * FROM telemetry ORDER BY id DESC LIMIT 300")
    if not metrics:
        st.info("Use the assistant and its timings will appear here.")
    else:
        df = pd.DataFrame(metrics)
        llm, tool, crit = (df[df.event_type == k] for k in ("llm", "tool", "critic"))
        errs = int((df.event_type == "llm_error").sum())
        m = st.columns(4)
        m[0].metric("AI requests", len(llm), border=True)
        m[1].metric("Median response", f"{llm.latency_ms.median() / 1000:.1f} s" if len(llm) else "n/a", border=True)
        m[2].metric("Tokens used", f"{int(llm.tokens_in.sum() + llm.tokens_out.sum()):,}", border=True)
        m[3].metric("Failed requests", errs, border=True)
        ok = df[df.event_type.isin(["llm", "tool", "critic", "stt"])]
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Average time per step (ms)**")
            st.bar_chart(ok.groupby("name").latency_ms.mean(), width="stretch")
        with c2:
            st.markdown("**Recent activity**")
            st.dataframe(df[["created_at", "event_type", "name", "provider", "latency_ms", "status"]].head(25)
                         .rename(columns={"created_at": "When", "event_type": "Type", "name": "Name", "provider": "Provider",
                                          "latency_ms": "ms", "status": "Result"}), hide_index=True, width="stretch")
