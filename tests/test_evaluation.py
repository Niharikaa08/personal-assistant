import hashlib

import numpy as np

import assistant.rag as rag
from assistant.config import load_settings
from assistant.db import Database
from assistant.evaluation import run_critic_eval, run_retrieval_eval, run_scheduler_eval, to_markdown
from assistant.llm import LLMUnavailable
from assistant.rag import Retriever
from assistant.tools import Toolbox


def test_scheduler_hard_constraints_hold_on_random_calendars():
    res = run_scheduler_eval(n=200, seed=5)
    assert res["total_violations"] == 0, res["violations"]


def test_priority_policy_serves_high_priority_work_sooner_than_fifo():
    p = run_scheduler_eval(n=200, seed=5)["policies"]
    assert p["priority"]["high_wait_h"] < p["fifo"]["high_wait_h"]
    assert p["priority"]["high_late_pct"] <= p["fifo"]["high_late_pct"]


def test_auto_schedule_does_not_book_the_same_task_twice(tmp_path):
    db = Database(tmp_path / "s.db")
    tb = Toolbox(db, Retriever(db), load_settings())
    for t in ("A", "B", "C"):
        db.execute("INSERT INTO tasks (title, priority) VALUES (?, 'high')", (t,))
    first = tb.auto_schedule_tasks(days_ahead=3)
    n = db.query("SELECT COUNT(*) AS n FROM events")[0]["n"]
    assert "Scheduled 3 focus blocks" in first and n == 3
    second = tb.auto_schedule_tasks(days_ahead=3)
    assert "already has focus time" in second
    assert db.query("SELECT COUNT(*) AS n FROM events")[0]["n"] == n
    db.execute("INSERT INTO tasks (title, priority) VALUES ('D', 'low')")      # new work still gets scheduled
    assert "Scheduled 1 focus blocks" in tb.auto_schedule_tasks(days_ahead=3)


class FakeEmbedder:
    """Hashed character trigrams: enough to exercise the dense and hybrid code paths offline."""
    def embed(self, texts):
        for t in texts:
            v = np.zeros(128, dtype=np.float32)
            t = t.lower()
            for i in range(len(t) - 2):
                v[int(hashlib.md5(t[i:i + 3].encode()).hexdigest(), 16) % 128] += 1
            yield v / (np.linalg.norm(v) or 1)


def test_retrieval_eval_sparse_only_when_no_embedding_model():
    r = run_retrieval_eval()                        # conftest disables the dense model
    assert r["dense_available"] is False and r["modes"]["dense"] is None and r["modes"]["hybrid"] is None
    assert r["modes"]["sparse"]["keyword"]["recall_at_k"] >= 75


def test_retrieval_eval_runs_all_three_methods_with_an_embedder(monkeypatch):
    monkeypatch.setattr(rag, "_EMBEDDER", FakeEmbedder())
    r = run_retrieval_eval()
    assert all(r["modes"][m] for m in ("sparse", "dense", "hybrid"))
    assert 0 <= r["modes"]["hybrid"]["all"]["mrr"] <= 1
    assert "Hybrid" in to_markdown({"retrieval": r})


class ConstLLM:
    def __init__(self, verdict):
        self.verdict = verdict

    def chat(self, *a, **k):
        if self.verdict is None:
            raise LLMUnavailable("down")
        return {"content": '{"score": %d, "verdict": "%s", "critique": "x", "suggested_fix": ""}'
                % (30 if self.verdict == "REVISE" else 95, self.verdict), "tool_calls": []}


def test_critic_eval_scoring_arithmetic():
    always_flag = run_critic_eval(ConstLLM("REVISE"))
    assert always_flag["interception_pct"] == 100 and always_flag["false_alarm_pct"] == 100
    never_flag = run_critic_eval(ConstLLM("PASS"))
    assert never_flag["interception_pct"] == 0 and never_flag["false_alarm_pct"] == 0
    broken = run_critic_eval(ConstLLM(None))
    assert broken["unverified"] == broken["faulty"] + broken["valid"] and broken["caught"] == 0
    md = to_markdown({"critic": always_flag})
    assert "Faulty actions flagged" in md
