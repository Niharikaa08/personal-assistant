"""Hybrid Document Retrieval (RAG): combines Dense Semantic Embeddings (FastEmbed)
and Sparse Lexical Search (TF-IDF) using Reciprocal Rank Fusion (RRF), plus lightweight
Knowledge Graph entity-relation extraction."""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

CHUNK_WORDS = 160
OVERLAP_WORDS = 30
MAX_SNIPPET_CHARS = 700
SUPPORTED = {".pdf", ".docx", ".txt", ".md"}

_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is False:
        return None
    if _EMBEDDER is None:
        try:
            from fastembed import TextEmbedding
            _EMBEDDER = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        except Exception:
            _EMBEDDER = False
            return None
    return _EMBEDDER


def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        import docx
        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs]
        for table in d.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(parts)
    if suffix in (".txt", ".md"):
        return data.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type '{suffix}'. Use: {', '.join(sorted(SUPPORTED))}")


def chunk_text(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = CHUNK_WORDS - OVERLAP_WORDS
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start : start + CHUNK_WORDS]
        chunks.append(" ".join(piece))
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


class Retriever:
    def __init__(self, db):
        self.db = db
        self._cache = None

    @property
    def mode(self) -> str:
        return "Hybrid (Dense BGE-Small + Sparse TF-IDF RRF)" if _get_embedder() else "Lexical (TF-IDF)"

    def _extract_doc_triples(self, filename: str, text: str) -> None:
        """Extract key structural relationships from uploaded documents into the Knowledge Graph."""
        self.db.add_edge("Knowledge Base", "INDEXED_DOC", filename, origin="rag")
        # Extract capitalized phrases or policy rules (e.g., 'Attendance', '75%', course codes)
        codes = set(re.findall(r"\b[A-Z]{2,4}[-\s]?\d{3}\b", text))
        for c in list(codes)[:8]:
            self.db.add_edge(filename, "REFERENCES_COURSE", c.strip(), origin="rag")
        pcts = re.findall(r"([A-Za-z][A-Za-z\s]{3,25})\s+(?:is|must be|minimum|at least)\s+(\d{1,3}%)", text, re.I)
        for subj, val in pcts[:6]:
            self.db.add_edge(subj.strip().title(), "THRESHOLD", val, origin="rag")

    def ingest(self, filename: str, data: bytes) -> int:
        text = extract_text(filename, data)
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No readable text found (scanned PDFs need OCR).")

        embedder = _get_embedder()
        embeddings = list(embedder.embed(chunks)) if embedder else [None] * len(chunks)

        self.db.execute("DELETE FROM chunks WHERE doc = ?", (filename,))
        for i, (ch, emb) in enumerate(zip(chunks, embeddings)):
            blob = np.asarray(emb, dtype=np.float32).tobytes() if emb is not None else None
            self.db.execute(
                "INSERT INTO chunks (doc, idx, text, embedding) VALUES (?, ?, ?, ?)",
                (filename, i, ch, blob),
            )
        self._extract_doc_triples(filename, text)
        self._cache = None
        return len(chunks)

    def list_docs(self) -> list[dict]:
        return self.db.query("SELECT doc AS name, COUNT(*) AS chunks FROM chunks GROUP BY doc ORDER BY doc")

    def delete_doc(self, name: str) -> None:
        self.db.execute("DELETE FROM chunks WHERE doc = ?", (name,))
        self._cache = None

    def _index(self):
        sig = self.db.query("SELECT COUNT(*) AS n, COALESCE(MAX(id), 0) AS m FROM chunks")[0]
        sig = (sig["n"], sig["m"])
        if self._cache and self._cache[0] == sig:
            return self._cache

        rows = self.db.query("SELECT doc, idx, text, embedding FROM chunks ORDER BY id")
        if not rows:
            self._cache = (sig, None, None, None, [])
            return self._cache

        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        tfidf_matrix = vec.fit_transform([r["text"] for r in rows])

        dense_matrix = None
        embedder = _get_embedder()
        if embedder and all(r["embedding"] is not None for r in rows):
            vecs = [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
            dense_matrix = np.vstack(vecs)

        self._cache = (sig, vec, tfidf_matrix, dense_matrix, rows)
        return self._cache

    def search(self, query: str, k: int = 3, mode: str = "hybrid") -> list[dict]:
        """mode: hybrid (default, RRF of both), sparse (TF-IDF only) or dense (embeddings only)."""
        _, vec, tfidf_matrix, dense_matrix, rows = self._index()
        if vec is None or not rows:
            return []

        n = len(rows)
        k = max(1, min(int(k), 6))

        try:
            q_sparse = vec.transform([query])
            sparse_scores = linear_kernel(q_sparse, tfidf_matrix).ravel()
        except ValueError:
            sparse_scores = np.zeros(n)

        embedder = _get_embedder() if mode != "sparse" else None
        if mode == "dense" and (embedder is None or dense_matrix is None):
            return []
        if embedder is not None and dense_matrix is not None:
            q_dense = next(embedder.embed([query]))
            dense_scores = dense_matrix @ np.asarray(q_dense, dtype=np.float32)

            sparse_ranks = np.argsort(np.argsort(-sparse_scores))
            dense_ranks = np.argsort(np.argsort(-dense_scores))
            rrf_scores = (1.0 / (60.0 + sparse_ranks)) + (1.0 / (60.0 + dense_ranks))
            order = np.argsort(-(dense_scores if mode == "dense" else rrf_scores))[:k]

            hits = []
            for i in order:
                if sparse_scores[i] <= 0.01 and dense_scores[i] < 0.42:
                    continue
                r = rows[i]
                hits.append({
                    "doc": r["doc"],
                    "idx": r["idx"],
                    "score": round(float(rrf_scores[i]), 4),
                    "dense_score": round(float(dense_scores[i]), 3),
                    "sparse_score": round(float(sparse_scores[i]), 3),
                    "text": r["text"][:MAX_SNIPPET_CHARS],
                })
            return hits

        order = sparse_scores.argsort()[::-1][:k]
        return [
            {
                "doc": rows[i]["doc"],
                "idx": rows[i]["idx"],
                "score": round(float(sparse_scores[i]), 4),
                "dense_score": 0.0,
                "sparse_score": round(float(sparse_scores[i]), 3),
                "text": rows[i]["text"][:MAX_SNIPPET_CHARS],
            }
            for i in order
            if sparse_scores[i] > 0.02
        ]
