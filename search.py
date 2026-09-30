"""Interactive demo: run one query through the pipeline and show the ranked snippets with timings.

Examples:
    python search.py --qid q6609                      # a query from the AppsRetrieval test split
    python search.py --text-file my_problem.txt        # your own problem statement
    python search.py --text "count pairs with sum divisible by k" --no-review
    python search.py --corpus my_code_v2.jsonl --text "..."   # your own corpus: one {"id","text"} per line
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

from acr.config import Settings
from acr.pipeline import AgenticRetriever


def load_apps():
    import mteb
    task = mteb.get_task("AppsRetrieval")
    task.load_data()
    split = task.dataset[next(iter(task.dataset))]["test"]
    corpus = split["corpus"]
    queries = dict(zip(split["queries"]["id"], split["queries"]["text"]))
    qrels = split["relevant_docs"]
    return list(corpus["id"]), list(corpus["text"]), queries, qrels


def load_jsonl(path: str):
    ids, texts = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            ids.append(str(row["id"]))
            texts.append(row["text"])
    return ids, texts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--qid", help="query id from the AppsRetrieval test split, e.g. q6609")
    src.add_argument("--text", help="query text")
    src.add_argument("--text-file", help="file containing the query text")
    p.add_argument("--corpus", help="JSONL corpus with one {\"id\", \"text\"} per line (default: AppsRetrieval)")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--no-review", action="store_true")
    p.add_argument("--show-lines", type=int, default=12, help="lines of code to show per result")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    settings = Settings()
    settings = replace(settings, use_llm=settings.use_llm and not args.no_llm, use_review=not args.no_review)

    gold = set()
    if args.corpus:
        doc_ids, doc_texts = load_jsonl(args.corpus)
        queries, qrels = {}, {}
    else:
        doc_ids, doc_texts, queries, qrels = load_apps()
    if args.qid:
        if args.qid not in queries:
            raise SystemExit(f"Unknown query id {args.qid}")
        query = queries[args.qid]
        gold = {d for d, s in qrels.get(args.qid, {}).items() if s > 0}
    else:
        query = args.text if args.text else Path(args.text_file).read_text(encoding="utf-8")

    t_load = time.perf_counter()
    retriever = AgenticRetriever(settings)
    t_model = time.perf_counter() - t_load
    retriever.index(doc_ids, doc_texts, show_progress=True)
    print(f"\nIndex: {len(doc_ids)} documents in {retriever.timings['index_s']:.1f}s "
          f"({retriever.timings['index_new_embeddings']} new embeddings, rest from cache)")

    t0 = time.perf_counter()
    result = retriever.search([query], top_k=args.top, show_progress=False)[0]
    t_total = time.perf_counter() - t0

    print("\n" + "=" * 100)
    print("QUERY:", " ".join(query.split())[:400], "..." if len(query) > 400 else "")
    if result.interpretation:
        print("\nLLM TASK:", result.interpretation.get("task", ""))
        print("LLM SKETCH:\n   " + result.interpretation.get("sketch", "").replace("\n", "\n   "))
    if settings.use_review:
        print(f"\nReview: {result.n_examples} example(s) found, {len(result.pool)} candidates run, "
              f"{len(result.passed)} passed")
    print("=" * 100)

    passed = set(result.passed)
    for rank, (d, score) in enumerate(result.ranking, 1):
        tags = []
        if d in passed:
            tags.append("PASSED EXAMPLES")
        if doc_ids[d] in gold:
            tags.append("CORRECT (ground truth)")
        print(f"\n#{rank}  {doc_ids[d]}  score={score:.4f}  {'  '.join(tags)}")
        lines = doc_texts[d].rstrip().split("\n")
        print("   " + "\n   ".join(lines[:args.show_lines]) + ("\n   ..." if len(lines) > args.show_lines else ""))

    if gold:
        ranks = [i for i, (d, _) in enumerate(result.ranking, 1) if doc_ids[d] in gold]
        print(f"\nCorrect solution rank: {ranks[0] if ranks else f'not in top {args.top}'}")
    t = retriever.timings
    print(f"\nTimings: model load {t_model:.1f}s | LLM {t.get('llm_s', 0):.2f}s | "
          f"embed query {t.get('embed_queries_s', 0):.2f}s | search {t.get('first_pass_s', 0):.3f}s | "
          f"review {t.get('review_s', 0):.2f}s | query total {t_total:.2f}s")


if __name__ == "__main__":
    main()
