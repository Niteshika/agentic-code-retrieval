"""MTEB wrappers.

- AgenticSearchModel: the full pipeline (sketch + original query + execution review). Uses MTEB's
  search interface (index/search), because the review needs to see the candidates.
- PrePostPipelineEncoder: the template-compatible encoder (AbsEncoder). Same pre-processing
  (AST-chunked documents, LLM-sketch-fused queries) but no review step.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

from .config import Settings
from .pipeline import AgenticRetriever


def build_meta(settings: Settings, mode: str, embed_dim: int) -> ModelMeta:
    tag = settings.tag(mode)
    return ModelMeta.create_empty(overwrites=dict(
        name=f"local/agentic-code-retrieval-{mode}",
        revision=tag,
        languages=["eng-Latn", "python-Code"],
        framework=["Sentence Transformers", "PyTorch"],
        similarity_fn_name=ScoringFunction.COSINE,
        embed_dim=embed_dim,
        max_tokens=settings.query_max_len,
        open_weights=True,
        use_instructions=False,
        reference=f"https://huggingface.co/{settings.embed_model}",
    ))


def _doc_text(row) -> str:
    title = (row.get("title") or "") if hasattr(row, "get") else ""
    text = row["text"]
    return f"{title} {text}".strip() if title else text


class AgenticSearchModel:
    """Implements MTEB's SearchProtocol (index + search)."""

    def __init__(self, settings: Settings | None = None, retriever: AgenticRetriever | None = None):
        self.settings = settings or Settings()
        self.retriever = retriever or AgenticRetriever(self.settings)
        self._meta = build_meta(self.settings, "agentic", self.retriever.embedder.dim)

    @property
    def mteb_model_meta(self) -> ModelMeta:
        return self._meta

    def index(self, corpus, *, task_metadata=None, hf_split=None, hf_subset=None,
              encode_kwargs=None, num_proc=None, **kwargs) -> None:
        ids, texts = [], []
        for row in corpus:
            ids.append(str(row["id"]))
            texts.append(_doc_text(row))
        self.retriever.index(ids, texts)

    def search(self, queries, *, task_metadata=None, hf_split=None, hf_subset=None, top_k: int = 1000,
               encode_kwargs=None, top_ranked=None, num_proc=None, **kwargs) -> dict[str, dict[str, float]]:
        qids = [str(row["id"]) for row in queries]
        texts = [row["text"] for row in queries]
        results = self.retriever.search(texts, top_k=top_k)
        ids = self.retriever.doc_ids
        return {qid: {ids[d]: score for d, score in r.ranking} for qid, r in zip(qids, results)}


class PrePostPipelineEncoder(AbsEncoder):
    """Template-compatible encoder: pre-processing only (no review)."""

    def __init__(self, settings: Settings | None = None, retriever: AgenticRetriever | None = None):
        self.settings = replace(settings or Settings(), use_review=False)
        self.retriever = retriever or AgenticRetriever(self.settings)
        self.mteb_model_meta = build_meta(self.settings, "encoder", self.retriever.embedder.dim)

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs) -> np.ndarray:
        texts = [t for batch in inputs for t in batch["text"]]
        if prompt_type == PromptType.query:
            return self.retriever.fused_query_vectors(texts)
        return self.retriever.encode_documents(texts)
