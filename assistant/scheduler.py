"""Constraint-based focus-time scheduler.

Pure functions with no database access, so the same code that serves the app can be
stress-tested with thousands of random calendars (see assistant/evaluation.py).

Hard constraints (never violated):
  * a block never overlaps an existing event or another block
  * blocks sit inside the daily window (09:00-21:00) and never in the past
  * consecutive blocks are separated by the requested break
Ordering policy (soft preference): which task gets the earliest free slot.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

DAY_START_HOUR, DAY_END_HOUR = 9, 21
PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}
POLICIES = ("priority", "deadline", "fifo")
NO_DUE = "9999-12-31"


@dataclass(frozen=True)
class Block:
    task_id: int
    title: str
    priority: str
    due: str | None
    start: datetime
    end: datetime


def order_tasks(tasks: list[dict], policy: str = "priority") -> list[dict]:
    """priority = high first then earliest due (production); deadline = earliest due first;
    fifo = creation order (naive baseline)."""
    rank = lambda t: PRIORITY_RANK.get(t.get("priority"), 1)
    due = lambda t: t.get("due") or NO_DUE
    keys = {
        "priority": lambda t: (rank(t), due(t), t["id"]),
        "deadline": lambda t: (due(t), rank(t), t["id"]),
        "fifo": lambda t: t["id"],
    }
    if policy not in keys:
        raise ValueError(f"unknown policy {policy!r}")
    return sorted(tasks, key=keys[policy])


def plan_blocks(tasks: list[dict], busy: list[tuple[datetime, datetime]], now: datetime, days_ahead: int = 3,
                session: timedelta = timedelta(minutes=60), buffer: timedelta = timedelta(minutes=15),
                policy: str = "priority") -> tuple[list[Block], list[dict]]:
    """Return (placed blocks, tasks that did not fit)."""
    queue = order_tasks(tasks, policy)
    blocks: list[Block] = []
    for d in range(days_ahead):
        if not queue:
            break
        day = (now + timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0)
        win_start, win_end = day.replace(hour=DAY_START_HOUR), day.replace(hour=DAY_END_HOUR)
        if d == 0 and now > win_start:
            win_start = now.replace(second=0, microsecond=0) + timedelta(minutes=(-now.minute) % 15)
        intervals = sorted((s, e) for s, e in busy if e > win_start and s < win_end)
        cursor = win_start
        for b_start, b_end in intervals + [(win_end, win_end)]:
            while queue and (b_start - cursor) >= session:
                t = queue.pop(0)
                end = cursor + session
                blocks.append(Block(t["id"], t["title"], t.get("priority", "medium"), t.get("due"), cursor, end))
                cursor = end + buffer
            cursor = max(cursor, b_end + buffer)
    return blocks, queue
