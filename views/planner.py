"""Planner: tasks and the coming week."""
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from assistant.tools import parse_dt
from ui.state import boot
from ui.theme import clean_title, day_label, esc, page_header, rel_day, section

ctx = boot()
db, agent, now = ctx.db, ctx.agent, ctx.now
FMT = "%Y-%m-%d %H:%M"

page_header("Planner", "Your tasks, and what the next seven days look like.")

tasks = db.query("SELECT id, title, priority, due, status FROM tasks ORDER BY status DESC, due IS NULL, due, id")
pending = [t for t in tasks if t["status"] == "pending"]
soon = (now + timedelta(days=2)).strftime("%Y-%m-%d 23:59")
nxt = db.query("SELECT title, start FROM events WHERE start >= ? ORDER BY start LIMIT 1", (now.strftime(FMT),))

m = st.columns(4)
m[0].metric("Open tasks", len(pending), border=True)
m[1].metric("Due within 2 days", sum(1 for t in pending if t["due"] and t["due"] <= soon), border=True)
m[2].metric("High priority", sum(1 for t in pending if t["priority"] == "high"), border=True)
if nxt:
    st_ = datetime.strptime(nxt[0]["start"], FMT)
    m[3].metric(f"Next event, {rel_day(st_.date(), now.date()).lower()}", f"{st_:%H:%M}", clean_title(nxt[0]["title"])[:34],
                delta_color="off", delta_arrow="off", border=True)
else:
    m[3].metric("Next event", "None", border=True)

left, right = st.columns([5, 4], gap="large")

# ------------------------------------------------------------------ tasks
with left:
    section("Tasks")
    with st.form("add_task", clear_on_submit=True, border=True):
        title = st.text_input("New task", placeholder="What needs doing?")
        c1, c2, c3 = st.columns(3)
        d = c1.date_input("Due date", value=None)
        t = c2.time_input("Time", value=None, step=900)
        prio = c3.selectbox("Priority", ["medium", "high", "low"])
        if st.form_submit_button("Add task", type="primary"):
            if not title.strip():
                st.warning("Give the task a name first.")
            else:
                due = f"{d} {(t or datetime.strptime('17:00', '%H:%M').time()):%H:%M}" if d else None
                st.toast(agent.tools.add_task(title.strip(), due=due, priority=prio))
                st.rerun()

    if tasks:
        ver = st.session_state.get("task_ver", 0)
        edited = st.data_editor(
            pd.DataFrame(tasks), hide_index=True, width="stretch", key=f"tasks_{ver}",
            column_order=("title", "priority", "due", "status"), disabled=["id"],
            column_config={
                "title": st.column_config.TextColumn("Task", width="medium", required=True),
                "priority": st.column_config.SelectboxColumn("Priority", options=["low", "medium", "high"], width=80),
                "due": st.column_config.TextColumn("Due", help="YYYY-MM-DD HH:MM", width=125),
                "status": st.column_config.SelectboxColumn("Status", options=["pending", "done"], width=80),
            })
        b1, b2, _ = st.columns([1.1, 1.3, 1])
        if b1.button("Save changes", width="stretch"):
            bad = []
            for _, row in edited.iterrows():
                due = row["due"] if isinstance(row["due"], str) and row["due"].strip() else None
                if due:
                    dt = parse_dt(due)
                    if not dt:
                        bad.append(row["title"])
                        continue
                    due = dt.strftime(FMT)
                db.execute("UPDATE tasks SET title=?, priority=?, status=?, due=? WHERE id=?",
                           (row["title"], row["priority"], row["status"], due, int(row["id"])))
            st.session_state.task_ver = ver + 1
            if bad:
                st.warning("Some dates were not saved (use YYYY-MM-DD HH:MM): " + ", ".join(bad))
            else:
                st.toast("Changes saved")
                st.rerun()
        with b2.popover("Auto-schedule", width="stretch"):
            st.caption("Places pending tasks into free time, highest priority first, with breaks in between.")
            days = st.slider("Days ahead", 1, 7, 3)
            mins = st.select_slider("Focus block", [30, 45, 60, 90, 120], value=60, format_func=lambda x: f"{x} min")
            brk = st.select_slider("Break", [0, 10, 15, 20, 30], value=15, format_func=lambda x: f"{x} min")
            if st.button("Schedule my tasks", type="primary", width="stretch"):
                st.session_state.sched_msg = agent.tools.auto_schedule_tasks(days, mins, brk)
                st.rerun()
        if st.session_state.get("sched_msg"):
            with st.container(border=True):
                st.markdown(st.session_state.pop("sched_msg").replace("\n", "  \n"))
    else:
        st.info("No tasks yet. Add one above, or ask the assistant.")

# ------------------------------------------------------------------ agenda
with right:
    section("Next seven days")
    start = now.replace(hour=0, minute=0)
    rows = db.query("SELECT title, start, end, notes FROM events WHERE start >= ? AND start < ? ORDER BY start",
                    (start.strftime(FMT), (start + timedelta(days=7)).strftime(FMT)))
    if not rows:
        st.info("Nothing scheduled. Use Auto-schedule to place your tasks into free time.")
    else:
        by_day = defaultdict(list)
        for r in rows:
            by_day[r["start"][:10]].append(r)
        html = ['<div class="agenda">']
        for day, evs in by_day.items():
            d0 = datetime.strptime(day, "%Y-%m-%d").date()
            html.append(f'<div class="day"><div class="day-h">{day_label(d0, now.date())}'
                        f'<small>{rel_day(d0, now.date())}</small></div>')
            for e in evs:
                focus = e["title"].startswith("[Focus]")
                name = clean_title(e["title"])
                html.append(f'<div class="row"><span class="time">{e["start"][11:]}–{e["end"][11:]}</span>'
                            f'<span>{esc(name)}{"<span class=tag>Focus</span>" if focus else ""}'
                            + (f'<div class="note">{esc(e["notes"])}</div>' if e["notes"] and not focus else "")
                            + '</span></div>')
            html.append("</div>")
        html.append("</div>")
        st.markdown("".join(html), unsafe_allow_html=True)
