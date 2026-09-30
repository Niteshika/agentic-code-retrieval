"""The full retrieval pipeline.

Documents:  AST chunking -> CodeRankEmbed per chunk -> length-weighted mean -> one vector per document
Queries:    LLM writes a code sketch -> score = (1 - w) * cos(original query, doc) + w * cos(sketch, doc)
Review:     candidates from the top-K of both searches are run on the problem's example tests;
            candidates that pass every example are moved to the top.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .chunker import ASTChunker
from .config import Settings
from .embedder import Embedder
from .llm import QueryInterpreter
from .review import Reviewer, parse_examples

log = logging.getLogger(__name__)

PASS_BONUS = 10.0     # cosine scores are <= 1, so passing candidates always end up above everything else
QUERY_BLOCK = 256     # queries scored together (limits memory for large query sets)


@dataclass
class QueryResult:
    ranking: list[tuple[int, float]]              # (document index, score), best first
    interpretation: dict | None = None             # the LLM's task / concepts / sketch
    pool: list[int] = field(default_factory=list)  # documents that were reviewed
    passed: list[int] = field(default_factory=list)
    n_examples: int = 0


class AgenticRetriever:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None,
                 interpreter: QueryInterpreter | None = None, reviewer: Reviewer | None = None):
        self.s = settings or Settings()
        self.embedder = embedder or Embedder(self.s)
        self.interpreter = interpreter or (QueryInterpreter(self.s) if self.s.use_llm else None)
        self.reviewer = reviewer or (Reviewer(self.s) if self.s.use_review else None)
        self.chunker = ASTChunker(max_chunk_chars=self.s.max_chunk_chars,
                                  add_ancestor_header=self.s.add_ancestor_header)
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self.D: np.ndarray | None = None
        self.timings: dict[str, float] = {}

    # ------------------------------------------------------------------ documents
    def encode_documents(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """One L2-normalized vector per document (pooled AST chunks, or the whole document)."""
        if not self.s.use_chunking:
            return self.embedder.embed(texts, "code", show_progress=show_progress)
        chunk_texts, owner, weights = [], [], []
        for i, text in enumerate(texts):
            for c in self.chunker.chunk(text):
                chunk_texts.append(c["text"])
                owner.append(i)
                weights.append(max(c["size"], 1))
        E = self.embedder.embed(chunk_texts, "code", show_progress=show_progress)
        out = np.zeros((len(texts), self.embedder.dim), dtype=np.float32)
        np.add.at(out, np.asarray(owner), E * np.asarray(weights, dtype=np.float32)[:, None])
        out /= np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)
        return out

    def index(self, doc_ids: list[str], doc_texts: list[str], show_progress: bool = True) -> None:
        t0 = time.perf_counter()
        before = dict(self.embedder.stats)
        self.doc_ids, self.doc_texts = list(doc_ids), list(doc_texts)
        self.D = self.encode_documents(self.doc_texts, show_progress=show_progress)
        self.timings["index_s"] = time.perf_counter() - t0
        self.timings["index_new_embeddings"] = self.embedder.stats["embedded"] - before["embedded"]
        log.info("Indexed %d documents in %.1fs (%d new embeddings, rest from cache)", len(doc_ids),
                 self.timings["index_s"], self.timings["index_new_embeddings"])

    # ------------------------------------------------------------------ queries
    def _query_side(self, query_texts: list[str], show_progress: bool = True):
        t0 = time.perf_counter()
        interps = (self.interpreter.interpret_many(query_texts, show_progress=show_progress)
                   if self.interpreter is not None else [None] * len(query_texts))
        t1 = time.perf_counter()
        O = self.embedder.embed(query_texts, "query", show_progress=show_progress)
        has_sketch = np.array([bool(it and it.get("sketch")) for it in interps])
        S = np.zeros_like(O)
        if has_sketch.any():
            S[has_sketch] = self.embedder.embed([it["sketch"] for it in interps if it and it.get("sketch")],
                                                "code", show_progress=show_progress)
        self.timings["llm_s"] = t1 - t0
        self.timings["embed_queries_s"] = time.perf_counter() - t1
        return O, S, has_sketch, interps

    def fused_query_vectors(self, query_texts: list[str], show_progress: bool = True) -> np.ndarray:
        """(1 - w) * original + w * sketch. Its cosine ranking equals the fused score ranking."""
        O, S, has_sketch, _ = self._query_side(query_texts, show_progress)
        w = self.s.sketch_weight
        Q = np.where(has_sketch[:, None], (1 - w) * O + w * S, O)
        return Q / np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12, None)

    # ------------------------------------------------------------------ search
    @staticmethod
    def _top_set(scores: np.ndarray, k: int) -> set[int]:
        if len(scores) <= k:
            return set(range(len(scores)))
        return set(np.argpartition(-scores, k)[:k].tolist())

    def search(self, query_texts: list[str], top_k: int = 10, show_progress: bool = True) -> list[QueryResult]:
        if self.D is None:
            raise RuntimeError("Call index() before search().")
        O, S, has_sketch, interps = self._query_side(query_texts, show_progress)
        w, k = self.s.sketch_weight, self.s.pool_k
        top_k = min(top_k, len(self.doc_ids))
        results: list[QueryResult] = []
        t_first, t_review = 0.0, 0.0

        for start in range(0, len(query_texts), QUERY_BLOCK):
            t0 = time.perf_counter()
            rows = range(start, min(start + QUERY_BLOCK, len(query_texts)))
            So = O[rows.start:rows.stop] @ self.D.T
            Ss = S[rows.start:rows.stop] @ self.D.T
            hs = has_sketch[rows.start:rows.stop]
            base = np.where(hs[:, None], (1 - w) * So + w * Ss, So)

            # ---- pools + review jobs ----
            block_info, jobs, job_owner = [], [], []
            for j, qi in enumerate(rows):
                examples = parse_examples(query_texts[qi]) if self.reviewer is not None else []
                pool: set[int] = set()
                if examples:
                    pool = self._top_set(So[j], k)
                    if hs[j]:
                        pool |= self._top_set(Ss[j], k)
                pool_list = sorted(pool)
                block_info.append((examples, pool_list))
                for d in pool_list:
                    jobs.append((self.doc_texts[d], examples))
                    job_owner.append((j, d))
            t1 = time.perf_counter()
            verdicts = self.reviewer.verdict_many(jobs, show_progress) if (self.reviewer and jobs) else []
            t2 = time.perf_counter()

            passed_by_q: dict[int, list[int]] = {}
            for (j, d), v in zip(job_owner, verdicts):
                if v == "pass":
                    passed_by_q.setdefault(j, []).append(d)

            for j, qi in enumerate(rows):
                final = base[j].copy()
                passed = passed_by_q.get(j, [])
                if passed:
                    final[passed] += PASS_BONUS
                top = np.argpartition(-final, top_k - 1)[:top_k] if top_k < len(final) else np.arange(len(final))
                top = top[np.argsort(-final[top])]
                examples, pool_list = block_info[j]
                results.append(QueryResult(
                    ranking=[(int(d), float(final[d])) for d in top],
                    interpretation=interps[qi], pool=pool_list, passed=sorted(passed),
                    n_examples=len(examples)))
            t_first += (t1 - t0) + (time.perf_counter() - t2)
            t_review += t2 - t1

        self.timings["first_pass_s"] = t_first
        self.timings["review_s"] = t_review
        return results
