"""All settings in one place. Values can be overridden with environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # ---------------- embedding model ----------------
    embed_model: str = "nomic-ai/CodeRankEmbed"
    query_prefix: str = "Represent this query for searching relevant code: "  # required by CodeRankEmbed
    query_max_len: int = 2048          # tokens; cuts only ~0.1% of AppsRetrieval queries
    doc_max_len: int = 1024            # tokens; cuts almost no chunks
    query_batch_size: int = 16
    doc_batch_size: int = 64
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))   # auto | cpu | cuda
    fp16_on_gpu: bool = True

    # ---------------- document pre-processing (AST chunking) ----------------
    use_chunking: bool = True
    max_chunk_chars: int = 512         # budget in non-whitespace characters (cAST paper's measure)
    add_ancestor_header: bool = True   # add "# def f(...):" above chunks cut out of a big function/class

    # ---------------- query pre-processing (LLM code sketch) ----------------
    use_llm: bool = True
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "gemini"))   # openai | gemini
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-5-mini"))
    openai_reasoning_effort: str = field(default_factory=lambda: _env("OPENAI_REASONING_EFFORT", "minimal"))
    openai_max_completion_tokens: int = 2000
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-3.8-flash"))
    gemini_max_output_tokens: int = 8000
    llm_workers: int = field(default_factory=lambda: int(_env("LLM_WORKERS", "8")))
    sketch_weight: float = 0.5         # query score = (1 - w) * original + w * sketch

    # ---------------- post-processing (execution-based review) ----------------
    use_review: bool = True
    pool_k: int = 20                   # review top-K of the sketch search + top-K of the original query
    review_timeout_s: float = 4.0      # per example run
    review_mem_mb: int = 1024          # memory limit per run (Linux/macOS only)
    review_workers: int = field(default_factory=lambda: int(_env("REVIEW_WORKERS", str(os.cpu_count() or 2))))
    python2_bin: str = field(default_factory=lambda: _env("PYTHON2_BIN", ""))   # optional, for Python 2 solutions
    float_tol: float = 1e-6

    # ---------------- paths ----------------
    cache_dir: Path = field(default_factory=lambda: Path(_env("CACHE_DIR", ".cache")))

    def __post_init__(self) -> None:
        if self.llm_provider not in ("openai", "gemini"):
            raise ValueError(f"LLM_PROVIDER must be 'openai' or 'gemini', got {self.llm_provider!r}")
        self.use_llm = self.use_llm and _env_bool("USE_LLM", True)
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def llm_model(self) -> str:
        return self.openai_model if self.llm_provider == "openai" else self.gemini_model

    def tag(self, mode: str) -> str:
        """Short name for this configuration (used as the MTEB model revision)."""
        parts = [mode, f"chunk{self.max_chunk_chars}" if self.use_chunking else "nochunk"]
        parts.append(f"{self.llm_provider}-{self.llm_model}" if self.use_llm else "nollm")
        if mode == "agentic":
            parts.append(f"review{self.pool_k}" if self.use_review else "noreview")
        return "_".join(parts)
