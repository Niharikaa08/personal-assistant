"""The Multi-Agent Orchestrator Loop:
Primary Executor Agent + Adversarial Critic Agent + Human-in-the-Loop Guardrails."""
from __future__ import annotations

import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from .critic import CriticAgent
from .llm import LLMUnavailable
from .tools import SENSITIVE

MAX_STEPS = 8
HISTORY_TURNS = 6

SYSTEM_PROMPT = """You are an intelligent personal assistant for {name}. Now: {now} ({weekday}), timezone {tz}.
Rules:
- Use tools for anything about tasks, calendar, reminders, documents, weather, web, email or memory. Never invent data.
- If a request needs 2+ steps, call make_plan first (short steps), then carry the steps out. If the user changes their mind or a step fails, call make_plan again with the revised plan.
- When asked to organize/time-block/optimize a schedule, use `auto_schedule_tasks`.
- Dates: 'YYYY-MM-DD HH:MM' 24h local time. Resolve words like 'tomorrow' / 'next Monday' from the current time above.
- Save durable personal facts or preferences the user states with remember_fact (not trivia).
- For questions about the user's documents call search_documents and cite sources like [file #chunk]. If nothing relevant is found, say so.
- send_email and delete/forget tools undergo an automatic Critic Audit and ask the user for approval - look up real emails/IDs from memory/calendar/docs BEFORE calling them.
- Be concise, structured, and helpful.
Known facts about the user:
{facts}"""


class Agent:
    def __init__(self, llm, toolbox, db, settings):
        self.llm, self.tools, self.db, self.s = llm, toolbox, db, settings
        self.critic = CriticAgent(llm, db)
        self.messages: list[dict] = []
        self.pending: dict | None = None
        self.plan: list[str] = []
        self.events: list[dict] = []
        self._queue: list[dict] = []
        self._steps = 0
        self._total_tokens = 0
        self._total_latency_ms = 0.0
        self._critic_retried = False

    def send(self, text: str) -> dict:
        self._abort_outstanding()
        self.messages.append({"role": "user", "content": text})
        self._trim_history()
        self._steps, self.events, self.plan = 0, [], []
        self._total_tokens, self._total_latency_ms = 0, 0.0
        self._critic_retried = False
        self.tools.last_plan = []
        return self._run()

    def resume(self, approved: bool) -> dict:
        if not self.pending:
            return self._result(reply="Nothing is waiting for approval.")
        call, self.pending = self.pending, None
        if approved:
            out = self._execute(call["id"], call["name"], call["raw_args"])
        else:
            out = "The user REJECTED this action. Do not retry it; acknowledge briefly and ask what to do instead."
            self.events.append({"tool": call["name"], "args": call["args"], "result": "🚫 rejected by user", "ms": 0})
        self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": out})
        return self._run()

    def _system_prompt(self) -> str:
        now = datetime.now(ZoneInfo(self.s.tz))
        rows = self.db.query("SELECT id, fact FROM facts ORDER BY id DESC LIMIT 30")
        facts = "\n".join(f"[{r['id']}] {r['fact']}" for r in rows) or "(none yet)"
        return SYSTEM_PROMPT.format(
            name=self.s.user_name,
            now=now.strftime("%Y-%m-%d %H:%M"),
            weekday=now.strftime("%A"),
            tz=self.s.tz,
            facts=facts,
        )

    def _trim_history(self) -> None:
        user_idx = [i for i, m in enumerate(self.messages) if m["role"] == "user"]
        if len(user_idx) > HISTORY_TURNS:
            self.messages = self.messages[user_idx[-HISTORY_TURNS]:]

    def _abort_outstanding(self) -> None:
        if self.pending:
            self.messages.append({
                "role": "tool",
                "tool_call_id": self.pending["id"],
                "content": "The user did not approve this action.",
            })
            self.pending = None
        for tc in self._queue:
            self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "Skipped."})
        self._queue = []

    def _execute(self, call_id: str, name: str, raw_args: str) -> str:
        t0 = time.perf_counter()
        result = self.tools.call(name, raw_args)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        if self.db:
            self.db.log_metric("tool", name, ms)
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except Exception:
            args = {}
        self.events.append({"tool": name, "args": args, "result": result, "ms": ms})
        if name == "make_plan" and self.tools.last_plan:
            self.plan = list(self.tools.last_plan)
        return result

    def _drain(self) -> bool:
        while self._queue:
            tc = self._queue.pop(0)
            if tc["name"] in SENSITIVE:
                try:
                    args = json.loads(tc["arguments"])
                except Exception:
                    args = {}

                # --- X-FACTOR #1: Adversarial Critic Audit before HITL Approval ---
                last_user_msg = next((m["content"] for m in reversed(self.messages) if m["role"] == "user"), "")
                recent_tools = "\n".join(f"{e['tool']}: {str(e['result'])[:250]}" for e in self.events[-3:])
                audit = self.critic.audit_action(last_user_msg, tc["name"], args, recent_tools)

                if audit.get("verdict") == "REVISE" and audit.get("score", 100) < 80 and not self._critic_retried:
                    self._critic_retried = True
                    self.events.append({
                        "tool": "🛡️ critic_audit",
                        "args": {"target": tc["name"], "score": audit["score"]},
                        "result": f"REVISE ({audit['score']}/100): {audit['critique']} -> {audit['suggested_fix']}",
                        "ms": audit.get("latency_ms", 0),
                    })
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": (
                            f"INTERNAL CRITIC REJECTION (Score {audit['score']}/100): {audit['critique']} "
                            f"Fix required: {audit['suggested_fix']}. Look up the right details or revise arguments and try again."
                        ),
                    })
                    for skipped in self._queue:
                        self.messages.append({"role": "tool", "tool_call_id": skipped["id"], "content": "Skipped for revision."})
                    self._queue = []
                    return False

                self.pending = {
                    "id": tc["id"],
                    "name": tc["name"],
                    "args": args,
                    "raw_args": tc["arguments"],
                    "audit": audit,
                }
                return True

            out = self._execute(tc["id"], tc["name"], tc["arguments"])
            self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
        return False

    def _run(self) -> dict:
        while True:
            if self._drain():
                return self._result(pending=self.pending)
            if self._steps >= MAX_STEPS:
                return self._result(reply="I hit my step limit for this request. Tell me how you'd like to continue.")
            self._steps += 1
            try:
                msg = self.llm.chat(
                    [{"role": "system", "content": self._system_prompt()}] + self.messages,
                    self.tools.schemas,
                )
                self._total_tokens += msg.get("tokens", 0)
                self._total_latency_ms += msg.get("latency_ms", 0.0)
            except LLMUnavailable as e:
                return self._result(
                    reply=("I couldn't reach an AI model just now.\n\n"
                           f"**What happened:** {e}\n\n"
                           "Open **Settings → AI connection** and run the connection test to see how to fix it."),
                    error=True,
                )
            assistant = {"role": "assistant", "content": msg["content"] or ""}
            if msg["tool_calls"]:
                assistant["tool_calls"] = [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]},
                        # Gemini 3.x needs its thought signature echoed back with the tool call
                        **({"extra_content": c["extra_content"]} if c.get("extra_content") else {}),
                    }
                    for c in msg["tool_calls"]
                ]
            self.messages.append(assistant)
            if not msg["tool_calls"]:
                return self._result(reply=msg["content"] or "(no response)")
            self._queue = list(msg["tool_calls"])

    def _result(self, reply=None, pending=None, error=False) -> dict:
        return {
            "reply": reply,
            "pending": pending,
            "events": list(self.events),
            "plan": list(self.plan),
            "error": error,
            "stats": {
                "steps": self._steps,
                "tokens": self._total_tokens,
                "latency_ms": round(self._total_latency_ms, 1),
            },
        }
