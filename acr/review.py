"""Post-processing: run candidate solutions on the example tests found in the problem statement.

A candidate that prints the expected output for every example is moved to the top of the ranking.
Verdicts are cached on disk by a hash of (code, examples), so no program is run twice.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm.auto import tqdm

from .config import Settings

try:                                   # memory limits only exist on Linux/macOS
    import resource
except ImportError:                    # Windows
    resource = None

SAMPLE_RE = re.compile(r"-----Sample Input[^\n]*?-----\n(.*?)\n-----Sample Output[^\n]*?-----\n(.*?)(?=\n-----|\Z)", re.S)
EXAMPLES_RE = re.compile(r"-----Examples?-----\n(.*?)(?=\n-----|\Z)", re.S)


def parse_examples(problem: str) -> list[tuple[str, str]]:
    """[(input, expected_output), ...] from an APPS-style problem statement, or [] if none can be read.

    Handles "-----Examples----- Input ... Output ..." (Codeforces style) and
    "-----Sample Input----- ... -----Sample Output-----" (AtCoder style).
    """
    # AtCoder style: an explanation can follow the output after a blank line, so keep only the first block
    exs = [(i.strip("\n") + "\n", o.strip().split("\n\n")[0].strip()) for i, o in SAMPLE_RE.findall(problem)]
    if exs:
        return exs
    m = EXAMPLES_RE.search(problem)
    if not m:
        return []
    parts = re.split(r"^(Input|Output)\s*$", m.group(1), flags=re.M)
    cur_in = None
    for tag, text in zip(parts[1::2], parts[2::2]):
        if tag == "Input":
            cur_in = text.strip("\n") + "\n"
        elif cur_in is not None:
            exs.append((cur_in, text.strip()))
            cur_in = None
    return [(i, o) for i, o in exs if o]


def outputs_match(got: str, expected: str, tol: float = 1e-6) -> bool:
    """Token-wise comparison: ignores whitespace, YES/yes, and tiny differences in decimal numbers."""
    g, e = got.split(), expected.split()
    if g == e:
        return True
    if len(g) != len(e):
        return False
    for a, b in zip(g, e):
        if a == b or a.lower() == b.lower():
            continue
        try:
            if not math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol):
                return False
        except ValueError:
            return False
    return True


def _is_python3(code: str) -> bool:
    try:
        compile(code, "<candidate>", "exec")
        return True
    except SyntaxError:
        return False
    except Exception:   # noqa: BLE001 - e.g. null bytes
        return False


class Reviewer:
    def __init__(self, settings: Settings):
        self.s = settings
        folder = settings.cache_dir / "review"
        folder.mkdir(parents=True, exist_ok=True)
        self.cache_path = folder / "verdicts.jsonl"
        self.cache: dict[str, str] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self.cache[rec["key"]] = rec["verdict"]
        self._lock = threading.Lock()

    def _limits(self):
        if resource is not None:
            lim = self.s.review_mem_mb * 2**20
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))

    @staticmethod
    def key(code: str, examples) -> str:
        return hashlib.sha256((code + "\x00" + json.dumps(examples)).encode("utf-8")).hexdigest()

    def run(self, code: str, examples) -> str:
        """'pass' if every example passes, else the first problem: 'wrong', 'error' or 'timeout'."""
        interpreter = sys.executable
        if self.s.python2_bin and not _is_python3(code):
            interpreter = self.s.python2_bin            # likely a Python 2 solution
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "solution.py"
            path.write_text(code, encoding="utf-8")
            for inp, expected in examples:
                try:
                    p = subprocess.run(
                        [interpreter, str(path)], input=inp, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=self.s.review_timeout_s, cwd=d,
                        preexec_fn=self._limits if (resource is not None and os.name == "posix") else None,
                    )
                except subprocess.TimeoutExpired:
                    return "timeout"
                except OSError:
                    return "error"
                if p.returncode != 0:
                    return "error"
                if not outputs_match(p.stdout, expected, self.s.float_tol):
                    return "wrong"
        return "pass"

    def verdict(self, code: str, examples) -> str:
        k = self.key(code, examples)
        if k in self.cache:
            return self.cache[k]
        v = self.run(code, examples)
        with self._lock:
            self.cache[k] = v
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": k, "verdict": v}) + "\n")
        return v

    def verdict_many(self, jobs: list[tuple[str, list]], show_progress: bool = True) -> list[str]:
        """jobs = [(code, examples), ...] -> verdicts in the same order. Runs uncached jobs in parallel."""
        with ThreadPoolExecutor(max_workers=self.s.review_workers) as pool:
            return list(tqdm(pool.map(lambda j: self.verdict(*j), jobs), total=len(jobs),
                             desc="Review (running candidates)", disable=not show_progress or not jobs))
