from acr.chunker import ASTChunker

SRC = '''import sys

class Solver:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]

    def run(self, s):
        dist = [float("inf")] * self.n
        dist[s] = 0
        for u in range(self.n):
            for v, w in self.adj[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
        return dist

n = int(input())
print(*Solver(n).run(0))
'''


def test_small_file_is_one_chunk():
    chunks = ASTChunker(max_chunk_chars=10_000).chunk(SRC)
    assert len(chunks) == 1
    assert chunks[0]["code"].strip() == SRC.strip()


def test_split_keeps_all_code_and_adds_headers():
    chunks = ASTChunker(max_chunk_chars=80).chunk(SRC)
    assert len(chunks) > 1
    all_text = "\n".join(c["text"] for c in chunks)
    for line in SRC.splitlines():                          # every line survives (signatures as headers)
        assert line.strip() in all_text
    assert any(c["text"].startswith("# class Solver:") for c in chunks)
    assert any("# def run(self, s):" in c["text"] for c in chunks)


def test_python2_and_broken_code_do_not_crash():
    for code in ["n = int(raw_input())\nprint n * 2\n", "def f(a):\n    return a +\n\nx = 5\n", ""]:
        chunks = ASTChunker(max_chunk_chars=5).chunk(code)
        assert len(chunks) >= 1
