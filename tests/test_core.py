"""Automated unit tests covering Hybrid RAG, CSP Auto-Scheduler, Critic Agent, and HITL Guardrails."""
import json
from assistant.config import load_settings
from assistant.db import Database
from assistant.rag import Retriever
from assistant.tools import Toolbox
from assistant.agent import Agent


class DummyLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.available = ["dummy"]

    def chat(self, messages, tools=None):
        if not self.responses:
            return {"content": "Done.", "tool_calls": [], "tokens": 10, "latency_ms": 5.0}
        return self.responses.pop(0)


def test_hybrid_rag_retrieval(tmp_path):
    db = Database(tmp_path / "test.db")
    ret = Retriever(db)
    content = b"Students must maintain 75% attendance in CS-402 to sit for final exams."
    chunks = ret.ingest("policy.txt", content)
    assert chunks == 1
    hits = ret.search("What is the attendance requirement for CS-402?", k=2)
    assert len(hits) == 1
    assert "75%" in hits[0]["text"]
    assert hits[0]["doc"] == "policy.txt"


def test_csp_auto_scheduler(tmp_path):
    s = load_settings()
    db = Database(tmp_path / "sched.db")
    tb = Toolbox(db, Retriever(db), s)
    tb.add_task("High Priority Thesis", priority="high")
    tb.add_task("Low Priority Reading", priority="low")
    out = tb.auto_schedule_tasks(days_ahead=2, session_minutes=60, break_minutes=15)
    assert "Scheduled 2 focus blocks" in out
    events = db.query("SELECT * FROM events ORDER BY start")
    assert len(events) == 2
    assert "High Priority Thesis" in events[0]["title"]


def test_critic_and_hitl_approval_flow(tmp_path):
    s = load_settings()
    db = Database(tmp_path / "hitl.db")
    tb = Toolbox(db, Retriever(db), s)
    tb.add_task("Temporary Task", priority="low")

    # 1. Primary LLM proposes deleting task #1
    # 2. Critic LLM audits and passes with score 95
    # 3. Primary LLM confirms deletion after user approves
    llm = DummyLLM([
        {"content": "", "tool_calls": [{"id": "c1", "name": "delete_task", "arguments": json.dumps({"task_id": 1})}], "tokens": 20, "latency_ms": 10},
        {"content": json.dumps({"score": 95, "verdict": "PASS", "critique": "Valid task ID.", "suggested_fix": ""}), "tool_calls": [], "tokens": 15, "latency_ms": 8},
        {"content": "Task #1 has been deleted.", "tool_calls": [], "tokens": 12, "latency_ms": 7},
    ])

    agent = Agent(llm, tb, db, s)
    res1 = agent.send("Delete task 1")
    assert res1["pending"] is not None
    assert res1["pending"]["name"] == "delete_task"
    assert res1["pending"]["audit"]["score"] == 95

    res2 = agent.resume(True)
    assert res2["pending"] is None
    assert "deleted" in res2["reply"].lower()
    assert db.query("SELECT COUNT(*) AS n FROM tasks")[0]["n"] == 0
