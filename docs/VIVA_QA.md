# 🎓 Final-Year Project Viva-Voce Q&A Guide

### Q1: Why use a Multi-Agent Adversarial Critic instead of a single LLM?
**Answer:** Single-pass LLM agents suffer from compounding hallucinations during function calling (e.g., guessing an email address or scheduling over an existing meeting). By decoupling generation from verification (`CriticAgent`), sensitive tool payloads are audited against ground-truth SQLite tables before reaching the user. If the audit score drops below 80/100, the Critic injects structured critique back into the message loop, forcing autonomous self-correction.

### Q2: Why Hybrid RAG with Reciprocal Rank Fusion (RRF) instead of pure vector search?
**Answer:** Dense embeddings (`BAAI/bge-small-en-v1.5`) capture semantic synonyms (e.g., *"absence rules"* matching *"attendance policy"*), whereas sparse `TF-IDF` excels at exact identifiers like course codes (`CS-402`) or percentages (`75%`). RRF (`1/(60 + rank_dense) + 1/(60 + rank_sparse)`) merges both rankings cleanly without requiring score calibration.

### Q3: Why offload scheduling to a deterministic algorithm (`auto_schedule_tasks`)?
**Answer:** Autoregressive transformers struggle with interval arithmetic and constraint satisfaction. Using the LLM for natural-language intent parsing and delegating time-slot allocation to a deterministic Python solver guarantees zero overlapping calendar blocks and enforces cognitive break buffers.

### Q4: How do you ensure safety when the agent can send emails or delete records?
**Answer:** Any tool listed in `SENSITIVE` pauses the execution state machine in `Agent._drain()`, persists the pending call + Critic score, and renders an explicit Human-in-the-Loop approval card in the UI. Nothing mutates external state without explicit user confirmation.

### Q5: How do you know the system actually works, rather than assuming it?
**Answer:** The Evaluation Lab (`assistant/evaluation.py`) turns each design claim into an experiment. The scheduler is stress-tested on random calendars and checked for overlap, window, past-time and break violations; three ordering policies are compared on deadlines. Retrieval is benchmarked as keyword vs semantic vs hybrid on keyword and paraphrased questions. The critic is scored on how many deliberately faulty actions it flags versus how often it raises a false alarm. Everything is seeded and exports a Markdown report. Limits: the corpus is small and synthetic, and the critic result depends on the chosen model.

### Q6: What happens when a model provider retires a model or fails?
**Answer:** `assistant/llm.py` does not trust hard-coded model names. It reads each provider's model list, tries a preference order, remembers retired models, and falls back to the other provider. Errors are classified (bad key, retired model, quota, rate limit, malformed tool call, network) and each gets the right response, e.g. retrying a malformed tool call with a different sample. Settings → Test connection runs the same checks on demand. This was added after the original defaults (`llama-3.3-70b-versatile`, `gemini-2.5-flash`) stopped working for new keys.
