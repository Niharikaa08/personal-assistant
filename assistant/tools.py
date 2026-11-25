"""Tools the agent can call (function calling), including the Algorithmic
Constraint-Satisfaction Auto-Scheduler and Knowledge Graph builder."""
from __future__ import annotations

import inspect
import json
import re
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
from zoneinfo import ZoneInfo

import requests

from .scheduler import plan_blocks

FMT = "%Y-%m-%d %H:%M"
DATE_ERR = "Invalid date/time. Use 'YYYY-MM-DD HH:MM' (24h) or 'YYYY-MM-DD'."
SENSITIVE = {"send_email", "delete_task", "delete_event", "forget_fact"}
MAX_TOOL_CHARS = 1800
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
UA = {"User-Agent": "ai-personal-assistant/2.0"}

WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain",
       65: "heavy rain", 71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
       81: "rain showers", 82: "violent showers", 95: "thunderstorm", 96: "thunderstorm + hail",
       99: "thunderstorm + hail"}


def parse_dt(value, end_of_day: bool = False):
    if not value:
        return None
    value = str(value).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt)
            if fmt == "%Y-%m-%d" and end_of_day:
                dt = dt.replace(hour=23, minute=59)
            return dt
        except ValueError:
            continue
    return None


def _fn(name, desc, props=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props or {}, "required": required or []}}}


_S = {"type": "string"}
_I = {"type": "integer"}

SCHEMAS = [
    _fn("make_plan", "Record a short step-by-step plan BEFORE doing a multi-step request. Call again to revise it.",
        {"steps": {"type": "array", "items": _S}}, ["steps"]),
    _fn("add_task", "Add a to-do. due='YYYY-MM-DD HH:MM'. priority low|medium|high. recurrence none|daily|weekly.",
        {"title": _S, "due": _S, "priority": _S, "recurrence": _S}, ["title"]),
    _fn("list_tasks", "List tasks. status pending|done.", {"status": _S}),
    _fn("complete_task", "Mark a task done by id (recurring tasks roll forward automatically).", {"task_id": _I}, ["task_id"]),
    _fn("delete_task", "Delete a task by id (needs user approval).", {"task_id": _I}, ["task_id"]),
    _fn("add_event", "Add a calendar event. start/end='YYYY-MM-DD HH:MM'. Warns on overlaps.",
        {"title": _S, "start": _S, "end": _S, "notes": _S}, ["title", "start"]),
    _fn("list_events", "List events between two dates (default: next 14 days).",
        {"date_from": _S, "date_to": _S}),
    _fn("delete_event", "Delete an event by id (needs user approval).", {"event_id": _I}, ["event_id"]),
    _fn("find_free_slots", "Find free time on a date (08:00-22:00) of at least N minutes.",
        {"date": _S, "duration_minutes": _I}, ["date"]),
    _fn("auto_schedule_tasks",
        "Algorithmically time-block all pending tasks into free calendar slots over the next N days "
        "based on priority, deadlines, and cognitive break buffers.",
        {"days_ahead": _I, "session_minutes": _I, "break_minutes": _I}),
    _fn("set_reminder", "Set a reminder. remind_at='YYYY-MM-DD HH:MM'.", {"text": _S, "remind_at": _S}, ["text", "remind_at"]),
    _fn("list_reminders", "List pending reminders."),
    _fn("remember_fact", "Save a durable fact/preference about the user for future chats.", {"fact": _S}, ["fact"]),
    _fn("forget_fact", "Delete a saved fact by id (needs user approval).", {"fact_id": _I}, ["fact_id"]),
    _fn("search_documents", "Search the user's uploaded documents; returns passages tagged [file #chunk].",
        {"query": _S, "k": _I}, ["query"]),
    _fn("web_search", "Search the web for current information.", {"query": _S}, ["query"]),
    _fn("get_weather", "Weather forecast for a city.", {"city": _S, "days": _I}, ["city"]),
    _fn("send_email", "Send an email (needs user approval).", {"to": _S, "subject": _S, "body": _S}, ["to", "subject", "body"]),
]
TOOL_NAMES = {s["function"]["name"] for s in SCHEMAS}


class Toolbox:
    def __init__(self, db, retriever, settings):
        self.db = db
        self.retriever = retriever
        self.s = settings
        self.last_plan: list[str] = []

    @property
    def schemas(self):
        return SCHEMAS

    def now(self) -> datetime:
        return datetime.now(ZoneInfo(self.s.tz)).replace(tzinfo=None)

    @staticmethod
    def _int(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return None

    def call(self, name: str, arguments) -> str:
        if name not in TOOL_NAMES:
            return f"Unknown tool '{name}'."
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
            if not isinstance(args, dict):
                raise ValueError
        except Exception:
            return "Tool error: arguments were not valid JSON."
        fn = getattr(self, name)
        allowed = inspect.signature(fn).parameters
        args = {k: v for k, v in args.items() if k in allowed and v is not None}
        try:
            out = fn(**args)
        except TypeError as e:
            return f"Tool error: bad arguments ({e})."
        except Exception as e:
            return f"Tool error: {type(e).__name__}: {e}"
        return str(out)[:MAX_TOOL_CHARS]

    def describe_action(self, name: str, args: dict) -> str:
        if name == "send_email":
            mode = "will be sent from your Gmail" if self._smtp_ready() else "preview only: it will be saved, not sent"
            quoted = "\n".join("> " + line for line in str(args.get("body") or "").splitlines())
            return (f"**Email**, {mode}\n\n**To:** `{args.get('to')}`\n\n**Subject:** {args.get('subject')}\n\n"
                    f"{quoted}")
        if name == "delete_task":
            r = self.db.query("SELECT title, due FROM tasks WHERE id = ?", (self._int(args.get("task_id")),))
            return f"**Delete task #{args.get('task_id')}:** {r[0]['title'] if r else '(not found)'}"
        if name == "delete_event":
            r = self.db.query("SELECT title, start FROM events WHERE id = ?", (self._int(args.get("event_id")),))
            return f"**Delete event #{args.get('event_id')}:** {r[0]['title'] + ' at ' + r[0]['start'] if r else '(not found)'}"
        if name == "forget_fact":
            r = self.db.query("SELECT fact FROM facts WHERE id = ?", (self._int(args.get("fact_id")),))
            return f"**Forget saved fact #{args.get('fact_id')}:** {r[0]['fact'] if r else '(not found)'}"
        return f"**{name}** {json.dumps(args)}"

    # ---- planning --------------------------------------------------------------------------
    def make_plan(self, steps):
        if isinstance(steps, str):
            steps = [steps]
        self.last_plan = [str(s)[:200] for s in steps][:8]
        return "Plan recorded. Now carry out step 1."

    # ---- tasks -----------------------------------------------------------------------------
    def add_task(self, title, due=None, priority="medium", recurrence="none"):
        due_n = None
        if due:
            dt = parse_dt(due)
            if not dt:
                return DATE_ERR
            due_n = dt.strftime(FMT)
        priority = priority if priority in ("low", "medium", "high") else "medium"
        recurrence = recurrence if recurrence in ("none", "daily", "weekly") else "none"
        if recurrence != "none" and not due_n:
            return "Recurring tasks need a due date."
        tid = self.db.execute("INSERT INTO tasks (title, due, priority, recurrence) VALUES (?,?,?,?)",
                              (title, due_n, priority, recurrence))
        self.db.add_edge(self.s.user_name, "HAS_TASK", title, origin="task")
        if due_n:
            self.db.add_edge(title, "DUE_ON", due_n[:10], origin="task")
        extra = (f", due {due_n}" if due_n else "") + (f", repeats {recurrence}" if recurrence != "none" else "")
        return f"Task #{tid} added: {title} [{priority}]{extra}"

    def list_tasks(self, status="pending"):
        status = "done" if status == "done" else "pending"
        rows = self.db.query("SELECT * FROM tasks WHERE status = ? ORDER BY due IS NULL, due, id", (status,))
        if not rows:
            return f"No {status} tasks."
        return "\n".join(
            f"#{r['id']} [{r['priority']}] {r['title']}"
            + (f" | due {r['due']}" if r["due"] else "")
            + (f" | repeats {r['recurrence']}" if r["recurrence"] != "none" else "")
            for r in rows)

    def complete_task(self, task_id):
        tid = self._int(task_id)
        rows = self.db.query("SELECT * FROM tasks WHERE id = ?", (tid,))
        if not rows:
            return f"No task #{task_id}."
        t = rows[0]
        self.db.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (tid,))
        msg = f"Task #{tid} marked done: {t['title']}."
        if t["recurrence"] != "none" and t["due"]:
            delta = timedelta(days=1 if t["recurrence"] == "daily" else 7)
            nxt, now = parse_dt(t["due"]) + delta, self.now()
            for _ in range(400):
                if nxt > now:
                    break
                nxt += delta
            new_id = self.db.execute("INSERT INTO tasks (title, due, priority, recurrence) VALUES (?,?,?,?)",
                                     (t["title"], nxt.strftime(FMT), t["priority"], t["recurrence"]))
            msg += f" Next {t['recurrence']} occurrence created as #{new_id}, due {nxt.strftime(FMT)}."
        return msg

    def delete_task(self, task_id):
        tid = self._int(task_id)
        if not self.db.query("SELECT 1 FROM tasks WHERE id = ?", (tid,)):
            return f"No task #{task_id}."
        self.db.execute("DELETE FROM tasks WHERE id = ?", (tid,))
        return f"Task #{tid} deleted."

    # ---- calendar & algorithmic CSP scheduler ----------------------------------------------
    def add_event(self, title, start, end=None, notes=""):
        s = parse_dt(start)
        if not s:
            return DATE_ERR
        e = parse_dt(end) if end else s + timedelta(hours=1)
        if not e:
            return DATE_ERR
        if e <= s:
            return "End must be after start."
        s_str, e_str = s.strftime(FMT), e.strftime(FMT)
        clashes = self.db.query("SELECT id, title, start, end FROM events WHERE start < ? AND end > ?", (e_str, s_str))
        eid = self.db.execute("INSERT INTO events (title, start, end, notes) VALUES (?,?,?,?)",
                              (title, s_str, e_str, notes or ""))
        self.db.add_edge(self.s.user_name, "SCHEDULED", title, origin="calendar")
        msg = f"Event #{eid} added: {title}, {s_str} to {e_str}."
        if clashes:
            msg += " WARNING - overlaps with: " + "; ".join(f"#{c['id']} {c['title']} ({c['start']}-{c['end']})" for c in clashes)
        return msg

    def list_events(self, date_from=None, date_to=None):
        start = parse_dt(date_from) or self.now().replace(hour=0, minute=0)
        end = parse_dt(date_to, end_of_day=True) or (start + timedelta(days=14))
        rows = self.db.query("SELECT * FROM events WHERE start >= ? AND start <= ? ORDER BY start",
                             (start.strftime(FMT), end.strftime(FMT)))
        if not rows:
            return f"No events between {start:%Y-%m-%d} and {end:%Y-%m-%d}."
        return "\n".join(f"#{r['id']} {r['start']} - {r['end'][11:]} {r['title']}" for r in rows)

    def delete_event(self, event_id):
        eid = self._int(event_id)
        if not self.db.query("SELECT 1 FROM events WHERE id = ?", (eid,)):
            return f"No event #{event_id}."
        self.db.execute("DELETE FROM events WHERE id = ?", (eid,))
        return f"Event #{eid} deleted."

    def find_free_slots(self, date, duration_minutes=60):
        day = parse_dt(date)
        if not day:
            return DATE_ERR
        dur = timedelta(minutes=self._int(duration_minutes) or 60)
        day = day.replace(hour=0, minute=0)
        win_start, win_end = day.replace(hour=8), day.replace(hour=22)
        now = self.now()
        if day.date() == now.date() and now > win_start:
            win_start = now.replace(second=0, microsecond=0) + timedelta(minutes=(-now.minute) % 15)
        rows = self.db.query("SELECT start, end FROM events WHERE end > ? AND start < ? ORDER BY start",
                             (win_start.strftime(FMT), win_end.strftime(FMT)))
        cursor, free = win_start, []
        for r in rows:
            s, e = parse_dt(r["start"]), parse_dt(r["end"])
            if s - cursor >= dur:
                free.append((cursor, s))
            cursor = max(cursor, e)
        if win_end - cursor >= dur:
            free.append((cursor, win_end))
        if not free:
            return f"No free slot of {int(dur.total_seconds() // 60)} min on {day:%Y-%m-%d} (08:00-22:00)."
        return f"Free on {day:%Y-%m-%d}: " + ", ".join(f"{a:%H:%M}-{b:%H:%M}" for a, b in free[:6])

    def auto_schedule_tasks(self, days_ahead=3, session_minutes=60, break_minutes=15):
        """Time-block pending tasks into free calendar slots (see assistant/scheduler.py)."""
        days = min(max(self._int(days_ahead) or 3, 1), 7)
        sess = timedelta(minutes=min(max(self._int(session_minutes) or 60, 30), 180))
        buf = timedelta(minutes=min(max(self._int(break_minutes) or 15, 0), 60))

        tasks = self.db.query("SELECT * FROM tasks WHERE status = 'pending'")
        if not tasks:
            return "No pending tasks to schedule."
        now = self.now()

        # Idempotent: a task that already has upcoming focus time is not booked a second time.
        booked = set()
        for r in self.db.query("SELECT notes FROM events WHERE notes LIKE 'Auto-scheduled for Task #%' AND end > ?",
                               (now.strftime(FMT),)):
            m = re.search(r"Task #(\d+)", r["notes"] or "")
            if m:
                booked.add(int(m.group(1)))
        todo = [t for t in tasks if t["id"] not in booked]
        if not todo:
            return "Every pending task already has focus time booked. Nothing new to schedule."

        busy = []
        for e in self.db.query("SELECT start, end FROM events WHERE end > ?", (now.strftime(FMT),)):
            s_, e_ = parse_dt(e["start"]), parse_dt(e["end"])
            if s_ and e_:
                busy.append((s_, e_))

        blocks, left = plan_blocks(todo, busy, now, days, sess, buf)
        if not blocks:
            return "No free time windows available to schedule pending tasks."
        lines = []
        for b in blocks:
            title = f"[Focus] {b.title} ({b.priority})"
            self.db.execute("INSERT INTO events (title, start, end, notes) VALUES (?,?,?,?)",
                            (title, b.start.strftime(FMT), b.end.strftime(FMT), f"Auto-scheduled for Task #{b.task_id}"))
            self.db.add_edge(title, "BLOCKS_TIME_FOR", b.title, origin="scheduler")
            lines.append(f"• {b.start:%a %d %b}, {b.start:%H:%M}–{b.end:%H:%M}: {b.title} ({b.priority})")
        rem = f"\n({len(left)} tasks deferred: the calendar is full)" if left else ""
        skipped = f"\n({len(booked)} already scheduled)" if booked else ""
        return f"Scheduled {len(blocks)} focus blocks:\n" + "\n".join(lines) + rem + skipped

    # ---- reminders -------------------------------------------------------------------------
    def set_reminder(self, text, remind_at):
        dt = parse_dt(remind_at)
        if not dt:
            return DATE_ERR
        rid = self.db.execute("INSERT INTO reminders (text, remind_at) VALUES (?, ?)", (text, dt.strftime(FMT)))
        return f"Reminder #{rid} set for {dt.strftime(FMT)}: {text}"

    def list_reminders(self):
        rows = self.db.query("SELECT * FROM reminders WHERE done = 0 ORDER BY remind_at")
        return "\n".join(f"#{r['id']} {r['remind_at']} {r['text']}" for r in rows) or "No pending reminders."

    # ---- long-term memory ------------------------------------------------------------------
    def remember_fact(self, fact):
        fact = str(fact).strip()[:300]
        if not fact:
            return "Nothing to remember."
        if self.db.query("SELECT 1 FROM facts WHERE lower(fact) = lower(?)", (fact,)):
            return "Already remembered."
        fid = self.db.execute("INSERT INTO facts (fact) VALUES (?)", (fact,))
        # Extract simple subject-relation-object triples for Knowledge Graph
        emails = EMAIL_RE.findall(fact)
        if emails:
            self.db.add_edge(self.s.user_name, "KNOWS_CONTACT", emails[0], origin="memory")
        self.db.add_edge(self.s.user_name, "REMEMBERS", fact[:45] + ("..." if len(fact) > 45 else ""), origin="memory")
        return f"Saved to long-term memory (#{fid}): {fact}"

    def forget_fact(self, fact_id):
        fid = self._int(fact_id)
        if not self.db.query("SELECT 1 FROM facts WHERE id = ?", (fid,)):
            return f"No fact #{fact_id}."
        self.db.execute("DELETE FROM facts WHERE id = ?", (fid,))
        return f"Fact #{fid} forgotten."

    # ---- retrieval / web / weather ---------------------------------------------------------
    def search_documents(self, query, k=3):
        hits = self.retriever.search(query, k or 3)
        if not hits:
            return "No relevant passages found in the uploaded documents."
        return "\n\n".join(f"[{h['doc']} #{h['idx']}] (RRF={h['score']}) {h['text']}" for h in hits)

    def web_search(self, query):
        lines = []
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            for r in DDGS().text(query, max_results=5):
                lines.append(f"- {r.get('title', '')}: {str(r.get('body', ''))[:220]} ({r.get('href', '')})")
        except Exception:
            lines = []
        if lines:
            return "\n".join(lines)
        try:
            r = requests.get("https://en.wikipedia.org/w/api.php", timeout=10, headers=UA,
                             params={"action": "query", "list": "search", "srsearch": query,
                                     "format": "json", "srlimit": 3})
            for item in r.json()["query"]["search"]:
                snippet = re.sub(r"<[^>]+>", "", item["snippet"])
                lines.append(f"- {item['title']}: {snippet} (https://en.wikipedia.org/wiki/{item['title'].replace(' ', '_')})")
        except Exception:
            pass
        return ("(Wikipedia fallback)\n" + "\n".join(lines)) if lines else "No web results found."

    def get_weather(self, city, days=3):
        geo = requests.get("https://geocoding-api.open-meteo.com/v1/search", timeout=10, headers=UA,
                           params={"name": city, "count": 1}).json()
        if not geo.get("results"):
            return f"Couldn't find a place called '{city}'."
        g = geo["results"][0]
        n = min(max(self._int(days) or 3, 1), 7)
        w = requests.get("https://api.open-meteo.com/v1/forecast", timeout=10, headers=UA, params={
            "latitude": g["latitude"], "longitude": g["longitude"], "timezone": "auto", "forecast_days": n,
            "current": "temperature_2m,weather_code",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"}).json()
        cur, d = w["current"], w["daily"]
        out = [f"{g['name']}, {g.get('country', '')}: now {cur['temperature_2m']}°C, {WMO.get(cur['weather_code'], 'unknown')}"]
        for i in range(len(d["time"])):
            out.append(f"{d['time'][i]}: {d['temperature_2m_min'][i]}-{d['temperature_2m_max'][i]}°C, "
                       f"{WMO.get(d['weather_code'][i], 'unknown')}, rain chance {d['precipitation_probability_max'][i]}%")
        return "\n".join(out)

    # ---- email -----------------------------------------------------------------------------
    def _smtp_ready(self) -> bool:
        return bool(self.s.gmail_address and self.s.gmail_app_password)

    def send_email(self, to, subject, body):
        if not EMAIL_RE.match(str(to).strip()):
            return f"'{to}' is not a valid email address."
        status = "dry-run (SMTP not configured, nothing was sent)"
        if self._smtp_ready():
            msg = EmailMessage()
            msg["From"], msg["To"], msg["Subject"] = self.s.gmail_address, to.strip(), subject
            msg.set_content(body)
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
                smtp.login(self.s.gmail_address, self.s.gmail_app_password)
                smtp.send_message(msg)
            status = "sent"
        self.db.execute("INSERT INTO outbox (to_addr, subject, body, status) VALUES (?,?,?,?)",
                        (to.strip(), subject, body, status))
        self.db.add_edge(self.s.user_name, "EMAILED", to.strip(), origin="email")
        return f"Email to {to}: {status}."
