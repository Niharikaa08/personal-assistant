"""Adversarial Critic Agent (X-Factor #1): audits sensitive tool calls against
ground-truth DB state, calendar conflicts, and user facts before Human-in-the-Loop approval.
If the audit score is < 80, forces the primary agent to self-correct."""
from __future__ import annotations

import json
import time

CRITIC_PROMPT = """You are a strict Adversarial Quality & Safety Auditor for an AI Personal Assistant.
Your job is to inspect a proposed sensitive tool action BEFORE it is presented to the user for approval.

Evaluate for:
1. Hallucinated email addresses, dates, task IDs, or event IDs not grounded in context.
2. Calendar scheduling conflicts or illogical times.
3. Missing recipient details, vague placeholders (like 'professor@example.com' when a real one exists), or unprofessional tone in emails.
4. Destructive deletions that contradict the user's stated goal.

Return ONLY valid JSON matching this exact schema:
{
  "score": <integer 0-100>,
  "verdict": "PASS" | "REVISE",
  "critique": "<1-2 sentence specific audit finding>",
  "suggested_fix": "<concrete fix if REVISE, else empty string>"
}"""


class CriticAgent:
    def __init__(self, llm, db):
        self.llm = llm
        self.db = db

    def audit_action(self, user_goal: str, tool_name: str, args: dict, recent_context: str) -> dict:
        t0 = time.perf_counter()
        facts = [r["fact"] for r in self.db.query("SELECT fact FROM facts ORDER BY id DESC LIMIT 15")]
        events = [
            f"#{r['id']} {r['title']} ({r['start']} to {r['end']})"
            for r in self.db.query("SELECT id, title, start, end FROM events ORDER BY start LIMIT 15")
        ]
        tasks = [
            f"#{r['id']} {r['title']} [{r['priority']}]"
            for r in self.db.query("SELECT id, title, priority FROM tasks WHERE status='pending' LIMIT 15")
        ]

        payload = (
            f"User Goal: {user_goal}\n"
            f"Proposed Action: {tool_name}({json.dumps(args, ensure_ascii=False)})\n"
            f"Known User Facts: {facts}\n"
            f"Pending Tasks: {tasks}\n"
            f"Current Calendar: {events}\n"
            f"Recent Tool Outputs:\n{recent_context[:1500]}"
        )

        try:
            resp = self.llm.chat(
                [
                    {"role": "system", "content": CRITIC_PROMPT},
                    {"role": "user", "content": payload},
                ],
                tools=None,
            )
            raw = (resp.get("content") or "").strip()
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("the reviewer did not return a verdict")
            verdict = json.loads(raw[start : end + 1])
            verdict["score"] = max(0, min(100, int(float(verdict.get("score", 0)))))
            verdict["verdict"] = "PASS" if str(verdict.get("verdict", "")).upper() == "PASS" else "REVISE"
            verdict.setdefault("critique", "")
            verdict.setdefault("suggested_fix", "")
        except Exception as e:
            # Fail visibly: an unreviewed action must never be shown as a passed review.
            verdict = {"score": None, "verdict": "UNVERIFIED",
                       "critique": f"Automatic safety review was unavailable ({str(e)[:120]}). Please check the details yourself.",
                       "suggested_fix": ""}

        ms = round((time.perf_counter() - t0) * 1000, 1)
        verdict["latency_ms"] = ms
        if self.db:
            self.db.log_metric("critic", f"audit:{tool_name}", ms, status=verdict.get("verdict", "PASS"))
        return verdict
