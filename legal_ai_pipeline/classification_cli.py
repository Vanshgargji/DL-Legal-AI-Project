"""Command line interface for Pipeline 2 classification."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .classification_config import ClassificationConfig
from .classification_features import extract_all_features
from .classification_labels import generate_weak_labels
from .classification_metrics import build_classification_analytics, evaluate_classification
from .classification_model import predict_all, train_classifier
from .classification_schema import create_label_schema
from .io_utils import ensure_dir, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AILA Pipeline 2 legal issue classification")
    parser.add_argument("--root", default=".", help="Project root containing data/processed")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["schema", "weak-labels", "features", "train", "predict", "evaluate", "analytics", "all"]:
        sub.add_parser(name)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ClassificationConfig.from_root(Path(args.root))
    ensure_dir(config.output_dir)
    started = time.time()
    if args.command == "schema":
        result = create_label_schema(config)
    elif args.command == "weak-labels":
        result = generate_weak_labels(config)
    elif args.command == "features":
        result = extract_all_features(config)
    elif args.command == "train":
        result = train_classifier(config)
    elif args.command == "predict":
        result = predict_all(config)
    elif args.command == "evaluate":
        result = evaluate_classification(config)
    elif args.command == "analytics":
        result = build_classification_analytics(config)
    elif args.command == "all":
        result = {
            "schema": create_label_schema(config),
            "weak_labels": generate_weak_labels(config),
            "features": extract_all_features(config),
            "model": train_classifier(config),
            "predictions": predict_all(config),
            "evaluation": evaluate_classification(config),
            "analytics": build_classification_analytics(config),
        }
    elapsed = time.time() - started
    manifest = {"command": args.command, "elapsed_seconds": elapsed, "result": result}
    write_json(config.output_dir / f"{args.command}_last_run.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
