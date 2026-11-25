"""SQLite persistence layer: tasks, events, reminders, facts, hybrid RAG chunks,
knowledge graph edges, outbox, and system telemetry."""
from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    due TEXT,
    priority TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'pending',
    recurrence TEXT DEFAULT 'none',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    start TEXT NOT NULL,
    end TEXT NOT NULL,
    notes TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    remind_at TEXT NOT NULL,
    done INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc TEXT NOT NULL,
    idx INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB
);
CREATE TABLE IF NOT EXISTS kg_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    relation TEXT NOT NULL,
    target TEXT NOT NULL,
    origin TEXT DEFAULT 'system',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    to_addr TEXT, subject TEXT, body TEXT, status TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    name TEXT NOT NULL,
    provider TEXT,
    latency_ms REAL,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    status TEXT DEFAULT 'ok',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""
TABLES = ["tasks", "events", "reminders", "facts", "chunks", "kg_edges", "outbox", "telemetry"]


class Database:
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with contextlib.closing(self._conn()) as c:
            c.executescript(SCHEMA)
            cols = [r[1] for r in c.execute("PRAGMA table_info(chunks)").fetchall()]
            if "embedding" not in cols:
                c.execute("ALTER TABLE chunks ADD COLUMN embedding BLOB")
            c.commit()

    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def execute(self, sql: str, params=()) -> int:
        with contextlib.closing(self._conn()) as c:
            cur = c.execute(sql, params)
            c.commit()
            return cur.lastrowid

    def query(self, sql: str, params=()) -> list[dict]:
        with contextlib.closing(self._conn()) as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    def log_metric(self, event_type: str, name: str, latency_ms: float,
                   provider: str = "", tokens_in: int = 0, tokens_out: int = 0, status: str = "ok") -> None:
        self.execute(
            "INSERT INTO telemetry (event_type, name, provider, latency_ms, tokens_in, tokens_out, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (event_type, name, provider, round(latency_ms, 1), tokens_in, tokens_out, status),
        )

    def add_edge(self, source: str, relation: str, target: str, origin: str = "memory") -> None:
        s, r, t = source.strip()[:80], relation.strip().upper()[:40], target.strip()[:80]
        if not (s and r and t):
            return
        exists = self.query(
            "SELECT 1 FROM kg_edges WHERE lower(source)=lower(?) AND upper(relation)=upper(?) AND lower(target)=lower(?)",
            (s, r, t),
        )
        if not exists:
            self.execute(
                "INSERT INTO kg_edges (source, relation, target, origin) VALUES (?, ?, ?, ?)",
                (s, r, t, origin),
            )

    def reset(self) -> None:
        with contextlib.closing(self._conn()) as c:
            for t in TABLES:
                c.execute(f"DELETE FROM {t}")
            c.commit()
