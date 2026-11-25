"""Evaluation Lab: reproducible experiments that measure the assistant's core claims.

  1. Scheduler   - stress test with random calendars: are the hard constraints ever broken, and
                   how do three ordering policies compare on deadlines?       (no AI needed)
  2. Retrieval   - sparse vs dense vs hybrid search on labelled queries.      (needs embedding model for dense)
  3. Critic      - does the safety reviewer flag faulty actions without crying wolf? (needs an AI provider)

Run from the command line to produce a report:   python -m assistant.evaluation --critic
or from the app: Insights -> Reliability lab.  Everything is seeded and repeatable.
"""
from __future__ import annotations

import random
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from .critic import CriticAgent
from .db import Database
from .eval_data import CORPUS, QUERIES, critic_cases
from .rag import Retriever, _get_embedder
from .scheduler import DAY_END_HOUR, DAY_START_HOUR, POLICIES, plan_blocks
from .seed import seed_demo

FMT = "%Y-%m-%d %H:%M"
FLAG_BELOW = 80          # same threshold the agent uses to force a self-correction


# ------------------------------------------------------------------ 1. scheduler
def _scenario(rng: random.Random):
    day0 = datetime(2026, 11, 24)
    now = day0 + timedelta(hours=rng.randint(5, 21), minutes=rng.choice(range(0, 60, 5)))
    busy = []
    for _ in range(rng.randint(0, 8)):
        start = day0 + timedelta(days=rng.randint(0, 4), hours=rng.randint(7, 20), minutes=rng.choice([0, 30]))
        busy.append((start, start + timedelta(minutes=rng.choice([30, 60, 90, 120, 180]))))
    tasks = []
    for i in range(rng.randint(3, 10)):
        due = None if rng.random() < 0.25 else (now + timedelta(hours=rng.randint(4, 96))).strftime(FMT)
        tasks.append({"id": i + 1, "title": f"Task {i + 1}", "due": due,
                      "priority": rng.choices(["high", "medium", "low"], [2, 3, 3])[0]})
    return (now, busy, tasks, timedelta(minutes=rng.choice([30, 45, 60, 90])),
            timedelta(minutes=rng.choice([0, 10, 15, 30])), rng.randint(1, 5))


def check_constraints(blocks, busy, now, session, buffer) -> dict:
    """Count violations of the hard constraints."""
    v = {"overlap": 0, "outside_window": 0, "in_the_past": 0, "short_break": 0, "wrong_length": 0}
    for i, b in enumerate(blocks):
        if any(b.start < e and s < b.end for s, e in busy) or any(
                b.start < o.end and o.start < b.end for o in blocks[i + 1:]):
            v["overlap"] += 1
        day = b.start.replace(hour=0, minute=0)
        if b.start < day.replace(hour=DAY_START_HOUR) or b.end > day.replace(hour=DAY_END_HOUR) or b.end.date() != b.start.date():
            v["outside_window"] += 1
        if b.start < now:
            v["in_the_past"] += 1
        if b.end - b.start != session:
            v["wrong_length"] += 1
    ordered = sorted(blocks, key=lambda b: b.start)
    for a, b in zip(ordered, ordered[1:]):
        if a.end.date() == b.start.date() and b.start - a.end < buffer and b.start >= a.end:
            v["short_break"] += 1
    return v


def run_scheduler_eval(n: int = 500, seed: int = 11) -> dict:
    rng = random.Random(seed)
    violations = {k: 0 for k in ("overlap", "outside_window", "in_the_past", "short_break", "wrong_length")}
    stats = {p: {"tasks": 0, "placed": 0, "late": 0, "high_placed": 0, "high_late": 0, "high_delay_h": 0.0}
             for p in POLICIES}
    for _ in range(n):
        now, busy, tasks, session, buffer, days = _scenario(rng)
        for policy in POLICIES:
            blocks, _left = plan_blocks(tasks, busy, now, days, session, buffer, policy)
            for k, c in check_constraints(blocks, busy, now, session, buffer).items():
                violations[k] += c
            s = stats[policy]
            s["tasks"] += len(tasks)
            s["placed"] += len(blocks)
            for b in blocks:
                late = bool(b.due) and b.end.strftime(FMT) > b.due
                s["late"] += late
                if b.priority == "high":
                    s["high_placed"] += 1
                    s["high_late"] += late
                    s["high_delay_h"] += (b.start - now).total_seconds() / 3600
    rows = {}
    for p, s in stats.items():
        rows[p] = {"placed_pct": 100 * s["placed"] / s["tasks"],
                   "late_pct": 100 * s["late"] / max(s["placed"], 1),
                   "high_late_pct": 100 * s["high_late"] / max(s["high_placed"], 1),
                   "high_wait_h": s["high_delay_h"] / max(s["high_placed"], 1)}
    return {"scenarios": n, "violations": violations, "total_violations": sum(violations.values()), "policies": rows}


# ------------------------------------------------------------------ 2. retrieval
def run_retrieval_eval(k: int = 3) -> dict:
    dense_ready = _get_embedder() is not None
    out = {"queries": len(QUERIES), "documents": len(CORPUS), "k": k, "dense_available": dense_ready, "modes": {}}
    with tempfile.TemporaryDirectory() as d:
        db = Database(Path(d) / "eval.db")
        retriever = Retriever(db)
        for name, text in CORPUS.items():
            retriever.ingest(name, text.encode())
        for mode in ("sparse", "dense", "hybrid"):
            if mode != "sparse" and not dense_ready:
                out["modes"][mode] = None
                continue
            ranks = {"keyword": [], "paraphrase": []}
            for query, expected, kind in QUERIES:
                docs = [h["doc"] for h in retriever.search(query, k=k, mode=mode)]
                ranks[kind].append(docs.index(expected) + 1 if expected in docs else None)

            def summarise(rs):
                n = len(rs)
                return {"recall_at_1": 100 * sum(r == 1 for r in rs) / n,
                        "recall_at_k": 100 * sum(r is not None for r in rs) / n,
                        "mrr": sum(1 / r for r in rs if r) / n}

            out["modes"][mode] = {"all": summarise(ranks["keyword"] + ranks["paraphrase"]),
                                  "keyword": summarise(ranks["keyword"]),
                                  "paraphrase": summarise(ranks["paraphrase"])}
    return out


# ------------------------------------------------------------------ 3. critic
def run_critic_eval(llm, tz: str = "Asia/Kolkata", trials: int = 1) -> dict:
    with tempfile.TemporaryDirectory() as d:
        db = Database(Path(d) / "critic.db")
        seed_demo(db, Retriever(db), tz)
        critic = CriticAgent(llm, db)
        rows, caught, false_alarm, unverified = [], 0, 0, 0
        faulty = valid = 0
        for label, goal, tool, args, is_faulty in critic_cases(db):
            for _ in range(trials):
                v = critic.audit_action(goal, tool, args, "")
                score = v.get("score")
                if v["verdict"] == "UNVERIFIED":
                    flagged, unverified = None, unverified + 1
                else:
                    flagged = v["verdict"] == "REVISE" or (score is not None and score < FLAG_BELOW)
                faulty += is_faulty
                valid += not is_faulty
                if flagged is not None:
                    caught += bool(is_faulty and flagged)
                    false_alarm += bool((not is_faulty) and flagged)
                rows.append({"case": label, "faulty": is_faulty, "flagged": flagged, "score": score,
                             "note": (v.get("critique") or "")[:160]})
    return {"trials": trials, "faulty": faulty, "valid": valid, "caught": caught, "false_alarms": false_alarm,
            "unverified": unverified, "interception_pct": 100 * caught / max(faulty, 1),
            "false_alarm_pct": 100 * false_alarm / max(valid, 1), "rows": rows}


# ------------------------------------------------------------------ report
def to_markdown(res: dict) -> str:
    L = ["# Evaluation report", "",
         "Generated by the built-in Evaluation Lab. All experiments are seeded and can be re-run from "
         "*Insights → Reliability lab* or with `python -m assistant.evaluation`.", ""]
    s = res.get("scheduler")
    if s:
        L += ["## 1. Scheduler stress test", "",
              f"{s['scenarios']} random calendars x 3 ordering policies. Hard constraints checked on every block: "
              "no overlaps, inside 09:00-21:00, not in the past, requested break respected.", "",
              f"**Constraint violations: {s['total_violations']}** ({', '.join(f'{k}: {v}' for k, v in s['violations'].items())})", "",
              "| Policy | Tasks placed | Placed after deadline | High-priority after deadline | High-priority wait (h) |",
              "|---|---|---|---|---|"]
        names = {"priority": "Priority, then deadline (used by the app)", "deadline": "Earliest deadline first",
                 "fifo": "Creation order (naive baseline)"}
        for p, r in s["policies"].items():
            L.append(f"| {names[p]} | {r['placed_pct']:.1f}% | {r['late_pct']:.1f}% | {r['high_late_pct']:.1f}% | {r['high_wait_h']:.1f} |")
        L.append("")
    r = res.get("retrieval")
    if r:
        L += ["## 2. Retrieval benchmark", "",
              f"{r['queries']} labelled queries over {r['documents']} short passages (synthetic corpus). "
              f"Recall@{r['k']} = right document in the top {r['k']}.", "",
              "| Method | Queries | Recall@1 | Recall@k | MRR |", "|---|---|---|---|---|"]
        for mode, label in (("sparse", "Keyword (TF-IDF)"), ("dense", "Semantic (embeddings)"), ("hybrid", "Hybrid (rank fusion)")):
            m = r["modes"].get(mode)
            if not m:
                L.append(f"| {label} | not run (embedding model unavailable) | | | |")
                continue
            for kind in ("all", "keyword", "paraphrase"):
                x = m[kind]
                L.append(f"| {label} | {kind} | {x['recall_at_1']:.0f}% | {x['recall_at_k']:.0f}% | {x['mrr']:.2f} |")
        L.append("")
    c = res.get("critic")
    if c:
        L += ["## 3. Safety reviewer (critic)", "",
              f"{c['faulty']} faulty and {c['valid']} correct actions. Without the reviewer, all faulty actions reach the "
              "user's approval screen unflagged.", "",
              f"- Faulty actions flagged: **{c['caught']}/{c['faulty']} ({c['interception_pct']:.0f}%)**",
              f"- Correct actions wrongly flagged: **{c['false_alarms']}/{c['valid']} ({c['false_alarm_pct']:.0f}%)**",
              f"- Reviews that could not run: {c['unverified']}", "",
              "| Case | Flagged | Score |", "|---|---|---|"]
        for row in c["rows"]:
            L.append(f"| {row['case']} | {'n/a' if row['flagged'] is None else 'yes' if row['flagged'] else 'no'} | {row['score']} |")
        L.append("")
    return "\n".join(L)


def main() -> None:
    import argparse

    from .config import load_settings
    from .llm import build_llm

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--critic", action="store_true", help="also run the critic benchmark (uses your AI provider)")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--out", default="docs/EVALUATION.md")
    a = ap.parse_args()
    results = {"scheduler": run_scheduler_eval(), "retrieval": run_retrieval_eval()}
    if a.critic:
        s = load_settings()
        results["critic"] = run_critic_eval(build_llm(s), s.tz, a.trials)
    md = to_markdown(results)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
