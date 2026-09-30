"""Run the pipeline on the CoIR AppsRetrieval test split with MTEB and write the submission JSON.

Examples:
    python run_mteb.py                          # full pipeline (sketch + review), writes appsretrieval_results.json
    python run_mteb.py --mode encoder           # template-compatible AbsEncoder (no review)
    python run_mteb.py --no-llm --no-review     # embedding-only baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from datetime import datetime

import mteb

from acr.config import Settings
from acr.mteb_models import AgenticSearchModel, PrePostPipelineEncoder


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["agentic", "encoder"], default="agentic",
                   help="agentic = full pipeline incl. review (MTEB search interface); "
                        "encoder = template AbsEncoder without review")
    p.add_argument("--provider", choices=["openai", "gemini"], help="overrides LLM_PROVIDER from .env")
    p.add_argument("--no-llm", action="store_true", help="skip the LLM code sketch")
    p.add_argument("--no-review", action="store_true", help="skip running candidates on the examples")
    p.add_argument("--no-chunking", action="store_true", help="embed whole documents instead of AST chunks")
    p.add_argument("--output", default="appsretrieval_results.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = Settings()
    overrides = {"use_llm": settings.use_llm and not args.no_llm,
                 "use_review": not args.no_review,
                 "use_chunking": not args.no_chunking}
    if args.provider:
        overrides["llm_provider"] = args.provider
    settings = replace(settings, **overrides)

    model = AgenticSearchModel(settings) if args.mode == "agentic" else PrePostPipelineEncoder(settings)
    print(f"Configuration: {settings.tag(args.mode)} | device: {model.retriever.embedder.device}")

    task = mteb.get_task("AppsRetrieval")
    t0 = time.perf_counter()
    result = mteb.evaluate(
        model,
        [task],
        encode_kwargs={"batch_size": 64},
        overwrite_strategy="always",
        prediction_folder=str(settings.cache_dir / "predictions" / settings.tag(args.mode)),
    )
    task_result = list(result.task_results)[0]

    result_dict = task_result.to_dict()
    if isinstance(result_dict.get("date"), datetime):          # json can't write datetime objects
        result_dict["date"] = result_dict["date"].timestamp()
    with open(args.output, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)

    scores = task_result.scores["test"][0]
    print(f"\nDone in {(time.perf_counter() - t0) / 60:.1f} min -> {args.output}")
    for key in ("ndcg_at_10", "mrr_at_10", "recall_at_1", "recall_at_10", "recall_at_100"):
        if key in scores:
            print(f"  {key:<14} {scores[key]:.4f}")
    interp = model.retriever.interpreter
    if interp is not None:
        print(f"  LLM usage this run: {interp.usage}")
    print(f"  Timings (s): { {k: round(v, 1) for k, v in model.retriever.timings.items()} }")


if __name__ == "__main__":
    main()
