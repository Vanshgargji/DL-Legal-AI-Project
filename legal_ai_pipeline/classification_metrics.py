"""Evaluation and analytics helpers for Pipeline 2."""

from __future__ import annotations

from collections import Counter, defaultdict

from .classification_config import ClassificationConfig
from .classification_schema import LABEL_SCHEMA
from .io_utils import ensure_dir, read_jsonl, write_csv, write_json


def _label_set(row: dict) -> set[str]:
    return {label["label"] for label in row.get("labels", [])}


def _load_label_map(path) -> dict[str, set[str]]:
    return {row["id"]: _label_set(row) for row in read_jsonl(path)}


def _score_predictions(pred_path, weak_path) -> dict:
    pred = _load_label_map(pred_path)
    weak = _load_label_map(weak_path)
    tp = fp = fn = 0
    per_label = {label: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for label in LABEL_SCHEMA}
    for item_id, gold in weak.items():
        guessed = pred.get(item_id, set())
        tp_set = guessed & gold
        fp_set = guessed - gold
        fn_set = gold - guessed
        tp += len(tp_set)
        fp += len(fp_set)
        fn += len(fn_set)
        for label in gold:
            per_label[label]["support"] += 1
        for label in tp_set:
            per_label[label]["tp"] += 1
        for label in fp_set:
            per_label[label]["fp"] += 1
        for label in fn_set:
            per_label[label]["fn"] += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"micro_precision": precision, "micro_recall": recall, "micro_f1": f1, "per_label": per_label}


def _qrels_agreement(config: ClassificationConfig) -> dict:
    query_labels = _load_label_map(config.predictions_dir / "query_predictions.jsonl")
    case_labels = _load_label_map(config.predictions_dir / "case_predictions.jsonl")
    statute_labels = _load_label_map(config.predictions_dir / "statute_predictions.jsonl")
    agreements = []
    for qrels_name, label_map in (("qrels_cases.jsonl", case_labels), ("qrels_statutes.jsonl", statute_labels)):
        for row in read_jsonl(config.processed_dir / qrels_name):
            if row["relevance"] <= 0:
                continue
            qset = query_labels.get(row["query_id"], set())
            dset = label_map.get(row["doc_id"], set())
            if qset or dset:
                agreements.append(1.0 if qset & dset else 0.0)
    return {"positive_qrels_checked": len(agreements), "label_agreement_rate": sum(agreements) / len(agreements) if agreements else 0.0}


def evaluate_classification(config: ClassificationConfig) -> dict:
    ensure_dir(config.metrics_dir)
    collections = {
        "cases": _score_predictions(config.predictions_dir / "case_predictions.jsonl", config.labels_dir / "weak_case_labels.jsonl"),
        "statutes": _score_predictions(config.predictions_dir / "statute_predictions.jsonl", config.labels_dir / "weak_statute_labels.jsonl"),
        "queries": _score_predictions(config.predictions_dir / "query_predictions.jsonl", config.labels_dir / "weak_query_labels.jsonl"),
    }
    per_label_rows = []
    for collection, result in collections.items():
        for label, stats in result["per_label"].items():
            precision = stats["tp"] / (stats["tp"] + stats["fp"]) if stats["tp"] + stats["fp"] else 0.0
            recall = stats["tp"] / (stats["tp"] + stats["fn"]) if stats["tp"] + stats["fn"] else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_label_rows.append(
                {
                    "collection": collection,
                    "label": label,
                    "support": stats["support"],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    agreement = _qrels_agreement(config)
    output = {"collections": collections, "qrels_label_agreement": agreement}
    write_json(config.metrics_dir / "classification_metrics.json", output)
    write_csv(config.metrics_dir / "per_label_metrics.csv", per_label_rows, ["collection", "label", "support", "precision", "recall", "f1"])
    low_conf = []
    for path_name in ("case_predictions.jsonl", "statute_predictions.jsonl", "query_predictions.jsonl"):
        for row in read_jsonl(config.predictions_dir / path_name):
            best = max((label["confidence"] for label in row["labels"]), default=0.0)
            if best < 0.45:
                low_conf.append({"id": row["id"], "item_type": row["item_type"], "best_confidence": best, "labels": ",".join(sorted(_label_set(row)))})
    write_json(config.metrics_dir / "error_analysis.json", {"low_confidence_count": len(low_conf), "examples": low_conf[:100]})
    return {"collections": {k: {kk: vv for kk, vv in v.items() if kk != "per_label"} for k, v in collections.items()}, "qrels_label_agreement": agreement}


def build_classification_analytics(config: ClassificationConfig) -> dict:
    ensure_dir(config.analytics_dir)
    distribution_rows = []
    examples = defaultdict(list)
    for file_name in ("case_predictions.jsonl", "statute_predictions.jsonl", "query_predictions.jsonl"):
        rows = read_jsonl(config.predictions_dir / file_name)
        collection = file_name.split("_", 1)[0]
        counts = Counter()
        for row in rows:
            for label in row["labels"]:
                counts[label["label"]] += 1
                if len(examples[(collection, label["label"])]) < 5:
                    examples[(collection, label["label"])].append(row["id"])
        for label in LABEL_SCHEMA:
            distribution_rows.append({"collection": collection, "label": label, "count": counts[label]})
    write_csv(config.analytics_dir / "label_distribution.csv", distribution_rows, ["collection", "label", "count"])
    low_conf_rows = []
    for path_name in ("case_predictions.jsonl", "statute_predictions.jsonl", "query_predictions.jsonl"):
        for row in read_jsonl(config.predictions_dir / path_name):
            best = max((label["confidence"] for label in row["labels"]), default=0.0)
            if best < 0.45:
                low_conf_rows.append({"id": row["id"], "item_type": row["item_type"], "best_confidence": best, "labels": ",".join(sorted(_label_set(row)))})
    write_csv(config.analytics_dir / "manual_review_candidates.csv", low_conf_rows, ["id", "item_type", "best_confidence", "labels"])

    def table_rows(collection: str) -> str:
        rows = [r for r in distribution_rows if r["collection"] == collection and r["count"] > 0]
        rows.sort(key=lambda r: (-r["count"], r["label"]))
        return "".join(f"<tr><td>{r['label']}</td><td>{r['count']}</td></tr>" for r in rows)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pipeline 2 Legal Issue Classification</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    th {{ background: #f8fafc; }}
    .note {{ color: #475569; }}
  </style>
</head>
<body>
  <h1>Pipeline 2: Legal Issue Classification</h1>
  <p class="note">Offline weak-supervision classifier over AILA cases, statutes, and queries.</p>
  <h2>Case Label Distribution</h2>
  <table><tr><th>Label</th><th>Count</th></tr>{table_rows('case')}</table>
  <h2>Statute Label Distribution</h2>
  <table><tr><th>Label</th><th>Count</th></tr>{table_rows('statute')}</table>
  <h2>Query Label Distribution</h2>
  <table><tr><th>Label</th><th>Count</th></tr>{table_rows('query')}</table>
  <p>Manual review candidates: {len(low_conf_rows)}</p>
</body>
</html>
"""
    (config.analytics_dir / "pipeline2_classification_report.html").write_text(html, encoding="utf-8")
    summary = {"distribution_rows": len(distribution_rows), "manual_review_candidates": len(low_conf_rows)}
    write_json(config.analytics_dir / "analytics_summary.json", summary)
    return summary
