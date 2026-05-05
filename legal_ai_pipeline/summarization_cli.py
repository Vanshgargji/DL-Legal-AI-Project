"""Command line interface for Pipeline 3 summarization."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .io_utils import ensure_dir, write_json
from .summarization_config import SummarizationConfig
from .summarization_pipeline import (
    build_summarization_analytics,
    evaluate_summaries,
    segment_all,
    summarize_cases,
    summarize_queries,
    summarize_statutes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AILA Pipeline 3 legal summarization")
    parser.add_argument("--root", default=".", help="Project root containing Pipeline 1 and 2 outputs")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["segment", "summarize-cases", "summarize-statutes", "summarize-queries", "evaluate", "analytics", "all"]:
        sub.add_parser(name)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SummarizationConfig.from_root(Path(args.root))
    ensure_dir(config.output_dir)
    started = time.time()
    if args.command == "segment":
        result = segment_all(config)
    elif args.command == "summarize-cases":
        result = summarize_cases(config)
    elif args.command == "summarize-statutes":
        result = summarize_statutes(config)
    elif args.command == "summarize-queries":
        result = summarize_queries(config)
    elif args.command == "evaluate":
        result = evaluate_summaries(config)
    elif args.command == "analytics":
        result = build_summarization_analytics(config)
    elif args.command == "all":
        result = {
            "segment": segment_all(config),
            "case_summaries": summarize_cases(config),
            "statute_summaries": summarize_statutes(config),
            "query_summaries": summarize_queries(config),
            "evaluation": evaluate_summaries(config),
            "analytics": build_summarization_analytics(config),
        }
    elapsed = time.time() - started
    manifest = {"command": args.command, "elapsed_seconds": elapsed, "result": result}
    write_json(config.output_dir / f"{args.command}_last_run.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
