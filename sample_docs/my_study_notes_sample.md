# CS-402 Distributed Systems & AI Agents — Study Notes

## Reciprocal Rank Fusion (RRF)
Hybrid search combines dense vector similarity (semantic meaning) with sparse lexical matching (TF-IDF/BM25).
Formula: `RRF(d) = 1 / (k + rank_dense(d)) + 1 / (k + rank_sparse(d))` where `k = 60`.

## Multi-Agent Reflection (Critic Pattern)
Single-pass LLM agents suffer from compounding tool-call errors. Introducing an Adversarial Critic Agent that audits proposed actions against ground-truth database state reduces hallucinated parameters before Human-in-the-Loop approval.
