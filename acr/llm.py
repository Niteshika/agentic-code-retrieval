"""Query pre-processing: an LLM (OpenAI or Gemini only) turns a problem statement into a code sketch.

Answers are cached on disk by a hash of (provider, model, query text), so every query is sent to the
API at most once. Failed calls are not cached and are retried on the next run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm

from .config import Settings

log = logging.getLogger(__name__)

N_CONCEPTS = 3
SYSTEM_PROMPT = f"""You help a code search engine find Python solutions to competitive programming problems.
The engine compares your text with the solutions' code, so the story in the problem (names, places, characters) does not help.

Read the problem and return ONLY a JSON object with these keys:
- "task": the problem restated in 1-2 plain sentences as a formal computational task, without the story.
- "concepts": a list of exactly {N_CONCEPTS} short search phrases (3-10 words each). Each should name a
  likely algorithm, technique or key operation of a solution (e.g. "binary search on sorted array",
  "count characters and take minimum ratio"). Make them different from each other.
- "sketch": a short Python solution sketch (at most 15 lines) in typical competitive programming style,
  reading input with input(). It does not have to be fully correct.

Return only the JSON, with no extra text."""


def parse_interpretation(text: str) -> dict:
    """Take the outermost {...} of the answer and check the fields."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the answer (possibly cut off)")
    obj = json.loads(text[start:end + 1])
    concepts = [str(c).strip() for c in obj.get("concepts", []) if str(c).strip()][:N_CONCEPTS]
    return {"task": str(obj.get("task", "")).strip(), "concepts": concepts,
            "sketch": str(obj.get("sketch", "")).strip()}


class QueryInterpreter:
    def __init__(self, settings: Settings):
        self.s = settings
        if settings.llm_provider == "openai":
            from openai import OpenAI
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not set (see .env.example)")
            self.client = OpenAI(api_key=key, max_retries=5)
        else:
            from google import genai
            from google.genai import types as genai_types
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set (see .env.example)")
            self.client = genai.Client(api_key=key)
            self._genai_types = genai_types

        folder = settings.cache_dir / "llm"
        folder.mkdir(parents=True, exist_ok=True)
        self.cache_path = folder / f"{settings.llm_provider}__{settings.llm_model}.jsonl"
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self.cache[rec["key"]] = rec
        self._lock = threading.Lock()
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "failures": 0}

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.s.llm_provider}|{self.s.llm_model}|{text}".encode("utf-8")).hexdigest()

    # ---- one API call: returns (text, input_tokens, output_tokens) ----
    def _call(self, prompt: str):
        if self.s.llm_provider == "openai":
            kwargs = dict(
                model=self.s.openai_model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                max_completion_tokens=self.s.openai_max_completion_tokens,
                response_format={"type": "json_object"},
            )
            if self.s.openai_reasoning_effort:
                kwargs["reasoning_effort"] = self.s.openai_reasoning_effort
            resp = self.client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            if not text:
                raise RuntimeError(f"empty answer (finish_reason={resp.choices[0].finish_reason})")
            return text, resp.usage.prompt_tokens, resp.usage.completion_tokens

        config = self._genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=self.s.gemini_max_output_tokens,
            response_mime_type="application/json",
        )
        for attempt in range(5):
            try:
                resp = self.client.models.generate_content(model=self.s.gemini_model, contents=prompt, config=config)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** (attempt + 1))
        text = resp.text or ""
        if not text:
            raise RuntimeError("empty answer")
        usage = resp.usage_metadata
        tin = getattr(usage, "prompt_token_count", 0) or 0
        tout = (getattr(usage, "candidates_token_count", 0) or 0) + (getattr(usage, "thoughts_token_count", 0) or 0)
        return text, tin, tout

    def interpret(self, text: str) -> dict | None:
        """Returns {"task", "concepts", "sketch"} or None if the LLM failed."""
        key = self._key(text)
        if key in self.cache:
            return self.cache[key]
        for attempt in range(2):                       # one extra try, e.g. for a cut-off JSON answer
            try:
                answer, tin, tout = self._call(text)
                rec = {"key": key, **parse_interpretation(answer)}
                with self._lock:
                    self.usage["calls"] += 1
                    self.usage["input_tokens"] += tin
                    self.usage["output_tokens"] += tout
                    self.cache[key] = rec
                    with open(self.cache_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                return rec
            except Exception as e:                     # noqa: BLE001 - never crash the whole run on one query
                log.warning("LLM failed (attempt %d): %s: %s", attempt + 1, type(e).__name__, str(e)[:200])
        with self._lock:
            self.usage["failures"] += 1
        return None

    def interpret_many(self, texts: list[str], show_progress: bool = True) -> list[dict | None]:
        todo = [t for t in dict.fromkeys(texts) if self._key(t) not in self.cache]
        if todo:
            with ThreadPoolExecutor(max_workers=self.s.llm_workers) as pool:
                list(tqdm(pool.map(self.interpret, todo), total=len(todo), desc="LLM sketches",
                          disable=not show_progress))
        return [self.cache.get(self._key(t)) for t in texts]
