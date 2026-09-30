"""cAST-style AST chunking with tree-sitter (split big nodes, merge small neighbours)."""

from __future__ import annotations

import tree_sitter_python as tspy
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspy.language())


def nonws_len(text: str) -> int:
    """Chunk size = number of non-whitespace characters (cAST paper's measure)."""
    return sum(1 for ch in text if not ch.isspace())


class ASTChunker:
    """Split a source file into chunks along syntax-tree boundaries.

    1. If the whole file fits the budget, it is one chunk.
    2. Otherwise top-level nodes are merged greedily until the next one would exceed the budget.
    3. A node that is too big on its own is split into its children; for functions/classes the
       signature is kept as an "ancestor header" on every chunk made from their body.
    Chunk text is always cut from the original bytes, so nothing is lost even if parsing hits an error.
    """

    def __init__(self, language=PY_LANGUAGE, max_chunk_chars: int = 512, add_ancestor_header: bool = True,
                 header_node_types=("function_definition", "class_definition")):
        self.parser = Parser(language)
        self.max_chars = max_chunk_chars
        self.add_ancestor_header = add_ancestor_header
        self.header_types = set(header_node_types)

    def chunk(self, code: str) -> list[dict]:
        src = code.encode("utf-8")
        root = self.parser.parse(src).root_node
        if nonws_len(code) <= self.max_chars or not root.children:
            groups = [([root], [])]
        else:
            groups = self._chunk_nodes(root.children, src, ancestors=[])
        chunks = [self._make_chunk(nodes, anc, src) for nodes, anc in groups]
        chunks = [c for c in chunks if c["code"].strip()] or [self._whole(code)]
        for i, c in enumerate(chunks):
            c["chunk_id"], c["n_chunks"], c["has_parse_error"] = i, len(chunks), root.has_error
        return chunks

    # ---- split-then-merge ----
    def _chunk_nodes(self, nodes, src, ancestors):
        groups, cur, cur_size = [], [], 0
        for node in nodes:
            size = nonws_len(self._text(node, src))
            if (not cur and size > self.max_chars) or (cur_size + size > self.max_chars):
                if cur:
                    groups.append((cur, ancestors))
                    cur, cur_size = [], 0
                if size > self.max_chars:
                    groups.extend(self._split_big_node(node, src, ancestors))
                    continue
            cur.append(node)
            cur_size += size
        if cur:
            groups.append((cur, ancestors))
        return groups

    def _split_big_node(self, node, src, ancestors):
        body = node.child_by_field_name("body")
        if body is not None and body.children:
            new_ancestors = ancestors
            if node.type in self.header_types:
                header = src[node.start_byte:body.start_byte].decode("utf-8", "replace")
                new_ancestors = ancestors + [" ".join(header.split())]
            return self._chunk_nodes(body.children, src, new_ancestors)
        if node.children:
            return self._chunk_nodes(node.children, src, ancestors)
        return [([node], ancestors)]          # a single huge token (e.g. a long string): keep it whole

    # ---- helpers ----
    @staticmethod
    def _text(node, src) -> str:
        return src[node.start_byte:node.end_byte].decode("utf-8", "replace")

    def _make_chunk(self, nodes, ancestors, src) -> dict:
        first, last = nodes[0], nodes[-1]
        raw = src[first.start_byte:last.end_byte].decode("utf-8", "replace")
        line_start = src.rfind(b"\n", 0, first.start_byte) + 1
        indent = src[line_start:first.start_byte].decode("utf-8", "replace")
        code = indent + raw if indent.strip() == "" else raw
        header = "".join(f"# {a}\n" for a in ancestors) if self.add_ancestor_header else ""
        return {
            "text": header + code,
            "code": code,
            "ancestors": list(ancestors),
            "start_line": first.start_point[0] + 1,
            "end_line": max(first.start_point[0] + 1, last.end_point[0] + (1 if last.end_point[1] > 0 else 0)),
            "size": nonws_len(code),
        }

    @staticmethod
    def _whole(code: str) -> dict:
        return {"text": code, "code": code, "ancestors": [], "start_line": 1,
                "end_line": max(code.count("\n"), 1), "size": nonws_len(code)}
