"""CodeRankEmbed wrapper with a content-addressed disk cache.

Every text is cached under a hash of its content, so re-indexing a new version of a codebase only
embeds the snippets (chunks) that actually changed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

from .config import Settings

log = logging.getLogger(__name__)


class EmbeddingCache:
    """Maps text-hash -> vector. Stored as keys.json + vectors.npy, written atomically."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.keys_path = self.folder / "keys.json"
        self.vec_path = self.folder / "vectors.npy"
        self.index: dict[str, int] = {}
        self.vectors: np.ndarray | None = None
        if self.keys_path.exists() and self.vec_path.exists():
            keys = json.loads(self.keys_path.read_text())
            vectors = np.load(self.vec_path)
            if len(keys) == len(vectors):
                self.index = {k: i for i, k in enumerate(keys)}
                self.vectors = vectors
            else:
                log.warning("Embedding cache in %s is inconsistent; starting a new one.", self.folder)

    def __len__(self) -> int:
        return len(self.index)

    def get(self, key: str) -> np.ndarray | None:
        i = self.index.get(key)
        return None if i is None else self.vectors[i]

    def add(self, keys: list[str], vectors: np.ndarray) -> None:
        new = [(k, v) for k, v in zip(keys, vectors) if k not in self.index]
        if not new:
            return
        start = len(self.index)
        block = np.stack([v for _, v in new]).astype(np.float32)
        self.vectors = block if self.vectors is None else np.concatenate([self.vectors, block])
        for j, (k, _) in enumerate(new):
            self.index[k] = start + j

    def save(self) -> None:
        if self.vectors is None:
            return
        keys = sorted(self.index, key=self.index.get)
        tmp_vec, tmp_keys = self.vec_path.with_suffix(".tmp.npy"), self.keys_path.with_suffix(".tmp")
        np.save(tmp_vec, self.vectors)
        tmp_keys.write_text(json.dumps(keys))
        os.replace(tmp_vec, self.vec_path)
        os.replace(tmp_keys, self.keys_path)


class Embedder:
    """Embeds natural-language queries ("query") and code ("code") with CodeRankEmbed."""

    SAVE_EVERY = 2048   # texts; the cache is saved after each block, so a crash loses at most one block

    def __init__(self, settings: Settings):
        import torch
        from sentence_transformers import SentenceTransformer

        self.s = settings
        if settings.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = settings.device
        log.info("Loading %s on %s", settings.embed_model, self.device)
        self.model = SentenceTransformer(settings.embed_model, trust_remote_code=True, device=self.device)
        if self.device == "cuda" and settings.fp16_on_gpu:
            self.model.half()
        self.dim = self._dimension()
        slug = settings.embed_model.replace("/", "__")
        self.cache = EmbeddingCache(settings.cache_dir / "embeddings" / slug)
        self._torch = torch
        self.stats = {"cache_hits": 0, "embedded": 0}

    def _dimension(self) -> int:
        getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return int(getter())

    def _key(self, text: str, kind: str) -> str:
        max_len = self.s.query_max_len if kind == "query" else self.s.doc_max_len
        prefix = self.s.query_prefix if kind == "query" else ""
        return hashlib.sha256(f"{kind}|{max_len}|{prefix}|{text}".encode("utf-8")).hexdigest()

    def embed(self, texts: list[str], kind: str, batch_size: int | None = None,
              show_progress: bool = False) -> np.ndarray:
        """kind = "query" (natural language, gets the prefix) or "code". Returns L2-normalized vectors."""
        assert kind in ("query", "code")
        keys = [self._key(t, kind) for t in texts]
        missing = list(dict.fromkeys(k for k in keys if self.cache.get(k) is None))
        self.stats["cache_hits"] += len(keys) - len(missing)
        if missing:
            text_of = dict(zip(keys, texts))
            is_query = kind == "query"
            self.model.max_seq_length = self.s.query_max_len if is_query else self.s.doc_max_len
            bs = batch_size or (self.s.query_batch_size if is_query else self.s.doc_batch_size)
            for start in range(0, len(missing), self.SAVE_EVERY):
                block = missing[start:start + self.SAVE_EVERY]
                block_texts = [(self.s.query_prefix if is_query else "") + text_of[k] for k in block]
                vecs = self.model.encode(block_texts, batch_size=bs, normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=show_progress)
                self.cache.add(block, vecs.astype(np.float32))
                self.cache.save()
                self.stats["embedded"] += len(block)
            if self.device == "cuda":
                self._torch.cuda.empty_cache()
        if not keys:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.cache.get(k) for k in keys]).astype(np.float32)
