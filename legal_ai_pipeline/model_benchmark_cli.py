"""CLI for the final model benchmarking pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .io_utils import ensure_dir, write_json
from .model_benchmark_config import ModelBenchmarkConfig
from .model_benchmark_pipeline import (
    build_model_report,
    environment_check,
    evaluate_model_runs,
    run_all,
    run_colbert_manifest,
    run_cross_encoder_rerank,
    run_dense_retrieval,
    run_model_rag,
    run_transformer_classification,
    run_transformer_summarization,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Final model benchmark pipeline")
    parser.add_argument("--root", default=".", help="Project root")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in [
        "check-env",
        "dense-retrieval",
        "cross-encoder",
        "classification",
        "colbert",
        "summarization",
        "rag",
        "evaluate",
        "analytics",
        "all",
    ]:
        sub.add_parser(name)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ModelBenchmarkConfig.from_root(Path(args.root))
    ensure_dir(config.output_dir)
    started = time.time()
    if args.command == "check-env":
        result = environment_check(config)
    elif args.command == "dense-retrieval":
        result = run_dense_retrieval(config)
    elif args.command == "cross-encoder":
        result = run_cross_encoder_rerank(config)
    elif args.command == "classification":
        result = run_transformer_classification(config)
    elif args.command == "colbert":
        result = run_colbert_manifest(config)
    elif args.command == "summarization":
        result = run_transformer_summarization(config)
    elif args.command == "rag":
        result = run_model_rag(config)
    elif args.command == "evaluate":
        result = evaluate_model_runs(config)
    elif args.command == "analytics":
        result = build_model_report(config)
    elif args.command == "all":
        result = run_all(config)
    elapsed = time.time() - started
    manifest = {"command": args.command, "elapsed_seconds": elapsed, "result": result}
    write_json(config.output_dir / f"{args.command}_last_run.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
