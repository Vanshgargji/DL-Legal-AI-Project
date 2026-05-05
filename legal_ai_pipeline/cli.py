"""Command line interface for Pipeline 1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .analytics import build_analytics
from .bm25 import run_bm25
from .colbert_fallback import run_colbert_fallback
from .config import PipelineConfig
from .data import ingest
from .doc2query import run_doc2query
from .io_utils import write_json
from .metrics import evaluate_run, evaluate_standard_runs
from .rag import run_rag


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AILA 2019 Pipeline 1 runner")
    parser.add_argument("--root", default=".", help="Project root containing the dataset aila folder")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["ingest", "bm25", "doc2query", "colbert", "evaluate", "rag", "analytics", "all"]:
        sub.add_parser(name)
    eval_one = sub.add_parser("evaluate-run")
    eval_one.add_argument("--run", required=True)
    eval_one.add_argument("--qrels", choices=["cases", "statutes"], required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PipelineConfig.from_root(Path(args.root))
    started = time.time()
    result = None
    if args.command == "ingest":
        result = ingest(config)
    elif args.command == "bm25":
        result = run_bm25(config)
    elif args.command == "doc2query":
        result = run_doc2query(config)
    elif args.command == "colbert":
        result = run_colbert_fallback(config)
    elif args.command == "evaluate":
        result = evaluate_standard_runs(config)
    elif args.command == "evaluate-run":
        qrels_path = config.processed_dir / ("qrels_cases.jsonl" if args.qrels == "cases" else "qrels_statutes.jsonl")
        run_path = Path(args.run)
        result = evaluate_run(run_path, qrels_path, config.metrics_dir / f"{run_path.stem}.json")
    elif args.command == "rag":
        result = run_rag(config)
    elif args.command == "analytics":
        result = build_analytics(config)
    elif args.command == "all":
        result = {
            "ingest": ingest(config),
            "bm25": run_bm25(config),
            "doc2query": run_doc2query(config),
            "colbert": run_colbert_fallback(config),
            "evaluate": evaluate_standard_runs(config),
            "rag": run_rag(config),
            "analytics": build_analytics(config),
        }
    elapsed = time.time() - started
    manifest = {"command": args.command, "elapsed_seconds": elapsed, "result": result}
    write_json(config.logs_dir / f"{args.command}_last_run.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
