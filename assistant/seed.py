"""Demo data seeder so the project can be demonstrated immediately."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


def seed_if_empty(db, retriever, tz: str) -> None:
    has_data = (
        db.query("SELECT COUNT(*) AS n FROM tasks")[0]["n"] > 0
        or db.query("SELECT COUNT(*) AS n FROM events")[0]["n"] > 0
        or db.query("SELECT COUNT(*) AS n FROM facts")[0]["n"] > 0
    )
    if not has_data:
        seed_demo(db, retriever, tz)


def seed_demo(db, retriever, tz: str) -> None:
    db.reset()
    now = datetime.now(ZoneInfo(tz)).replace(tzinfo=None)
    today = now.strftime("%Y-%m-%d")
    tmrw = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    day2 = (now + timedelta(days=2)).strftime("%Y-%m-%d")

    # Facts
    facts = [
        "My academic advisor is Dr. Ananya Sharma (ananya.sharma@university.edu).",
        "I am a final-year Computer Science engineering student based in Bhubaneswar.",
        "I prefer deep-work study sessions in the morning before 13:00.",
    ]
    for f in facts:
        db.execute("INSERT INTO facts (fact) VALUES (?)", (f,))
    db.add_edge("Student", "ADVISED_BY", "Dr. Ananya Sharma", origin="seed")
    db.add_edge("Dr. Ananya Sharma", "EMAIL", "ananya.sharma@university.edu", origin="seed")

    # Tasks
    db.execute("INSERT INTO tasks (title, due, priority, recurrence) VALUES (?,?,?,?)",
               ("Submit Final Year Capstone Architecture Report", f"{tmrw} 17:00", "high", "none"))
    db.execute("INSERT INTO tasks (title, due, priority, recurrence) VALUES (?,?,?,?)",
               ("Review Distributed Systems (CS-402) Lecture Notes", f"{day2} 14:00", "high", "none"))
    db.execute("INSERT INTO tasks (title, due, priority, recurrence) VALUES (?,?,?,?)",
               ("Prepare Viva Presentation Slides", f"{day2} 20:00", "medium", "none"))
    db.execute("INSERT INTO tasks (title, due, priority, recurrence) VALUES (?,?,?,?)",
               ("Daily LeetCode Practice (2 Graph Problems)", f"{today} 21:00", "low", "daily"))

    # Events
    db.execute("INSERT INTO events (title, start, end, notes) VALUES (?,?,?,?)",
               ("Compiler Design Lab", f"{tmrw} 10:00", f"{tmrw} 12:00", "Room Lab-3"))
    db.execute("INSERT INTO events (title, start, end, notes) VALUES (?,?,?,?)",
               ("Capstone Project Sync with Dr. Sharma", f"{tmrw} 15:00", f"{tmrw} 16:00", "Faculty Block"))

    # Reminder
    due_soon = (now + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M")
    db.execute("INSERT INTO reminders (text, remind_at) VALUES (?,?)",
               ("Upload signed project synopsis form", due_soon))

    # Sample documents
    sample_dir = Path(__file__).resolve().parent.parent / "sample_docs"
    for p in sample_dir.glob("*"):
        if p.is_file():
            try:
                retriever.ingest(p.name, p.read_bytes())
            except Exception:
                pass
