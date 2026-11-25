"""Insights -> Reliability lab: run the benchmarks and read the results."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from assistant.evaluation import run_critic_eval, run_retrieval_eval, run_scheduler_eval, to_markdown

POLICY_NAMES = {"priority": "Priority, then deadline (used by the app)",
                "deadline": "Earliest deadline first",
                "fifo": "Creation order (naive baseline)"}
METHOD_NAMES = {"sparse": "Keyword search", "dense": "Semantic search", "hybrid": "Hybrid (both combined)"}


def _store() -> dict:
    return st.session_state.setdefault("eval_results", {})


def render(ctx) -> None:
    res = _store()
    st.markdown("Rather than just claiming the assistant is reliable, measure it. Each experiment is seeded, "
                "so the numbers are repeatable.")

    # ---------------------------------------------------------------- scheduler
    with st.container(border=True):
        st.markdown("**1. Scheduler stress test**")
        st.caption("Builds hundreds of random calendars and checks that no focus block ever overlaps an event, "
                   "leaves the 09:00 to 21:00 window, starts in the past or ignores your break.")
        if st.button("Run stress test", key="ev_sched"):
            with st.spinner("Scheduling 500 random weeks…"):
                res["scheduler"] = run_scheduler_eval(n=500)
        s = res.get("scheduler")
        if s:
            a, b = st.columns(2)
            a.metric("Rule violations", s["total_violations"], border=True)
            b.metric("Random calendars tested", s["scenarios"], border=True)
            st.dataframe(pd.DataFrame([{
                "Ordering policy": POLICY_NAMES[p], "Tasks placed": f"{r['placed_pct']:.0f}%",
                "Placed after deadline": f"{r['late_pct']:.1f}%",
                "High-priority after deadline": f"{r['high_late_pct']:.1f}%",
                "High-priority wait (hours)": round(r["high_wait_h"], 1)} for p, r in s["policies"].items()]),
                hide_index=True, width="stretch")
            st.caption("Reading it: every policy obeys the rules. They differ in who gets the earliest slot, "
                       "which is a trade-off between serving high-priority work sooner and missing fewer deadlines overall.")

    # ---------------------------------------------------------------- retrieval
    with st.container(border=True):
        st.markdown("**2. Document search benchmark**")
        st.caption("Asks 18 questions against 16 short policy passages (a synthetic set). Half share keywords with the answer, "
                   "half are paraphrases. Compares keyword search, semantic search and the hybrid the app uses.")
        if st.button("Run benchmark", key="ev_rag"):
            with st.spinner("Indexing passages and searching…"):
                res["retrieval"] = run_retrieval_eval()
        r = res.get("retrieval")
        if r:
            rows = []
            for mode, label in METHOD_NAMES.items():
                m = r["modes"].get(mode)
                if m:
                    rows.append({"Method": label,
                                 "Keyword questions": f"{m['keyword']['recall_at_1']:.0f}%",
                                 "Paraphrased questions": f"{m['paraphrase']['recall_at_1']:.0f}%",
                                 "Overall, in top 3": f"{m['all']['recall_at_k']:.0f}%",
                                 "Ranking quality (MRR)": round(m["all"]["mrr"], 2)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("The two question columns show how often the right document came up first.")
            if not r["dense_available"]:
                st.info("The semantic search model isn't loaded here, so only keyword results are shown. "
                        "It downloads automatically on first use when the app has internet access.")

    # ---------------------------------------------------------------- critic
    with st.container(border=True):
        st.markdown("**3. Safety reviewer test**")
        st.caption("Feeds the reviewer 5 faulty actions (invented email address, wrong deletion, claims of unfinished work, "
                   "and so on) and 5 correct ones. Measures how many faulty actions it flags and how often it raises a false alarm.")
        if st.button("Run safety test", key="ev_critic", disabled=not ctx.agent.llm.available,
                     help=None if ctx.agent.llm.available else "Connect an AI provider in Settings first"):
            with st.spinner("Reviewing 10 actions…"):
                res["critic"] = run_critic_eval(ctx.agent.llm, ctx.settings.tz)
        c = res.get("critic")
        if c:
            a, b, d = st.columns(3)
            a.metric("Faulty actions flagged", f"{c['caught']} of {c['faulty']}", border=True)
            b.metric("False alarms", f"{c['false_alarms']} of {c['valid']}", border=True)
            d.metric("Reviews that failed to run", c["unverified"], border=True)
            st.dataframe(pd.DataFrame([{"Case": x["case"], "Flagged": "n/a" if x["flagged"] is None else ("Yes" if x["flagged"] else "No"),
                                        "Score": x["score"], "Reviewer's note": x["note"]} for x in c["rows"]]),
                         hide_index=True, width="stretch")
            if c["unverified"]:
                st.warning("Some reviews could not run, so these numbers are incomplete. Check Settings → AI connection.")

    if res:
        st.download_button("Download report (Markdown)", to_markdown(res), file_name="evaluation_report.md",
                           mime="text/markdown", icon=":material/download:")
