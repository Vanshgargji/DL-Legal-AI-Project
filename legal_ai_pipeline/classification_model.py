"""Rule-assisted legal issue classifier."""

from __future__ import annotations

from collections import Counter, defaultdict

from .classification_config import ClassificationConfig
from .classification_labels import _labels_from_scores
from .classification_schema import LABEL_SCHEMA
from .io_utils import ensure_dir, read_jsonl, write_json, write_jsonl


def train_classifier(config: ClassificationConfig) -> dict:
    ensure_dir(config.models_dir)
    train_rows = read_jsonl(config.labels_dir / "train_dataset.jsonl")
    label_support = Counter()
    for row in train_rows:
        for label in row["labels"]:
            label_support[label["label"]] += 1
    model = {
        "model_type": "rule_assisted_keyword_classifier",
        "multi_label": True,
        "threshold": config.confidence_threshold,
        "max_labels_per_item": config.max_labels_per_item,
        "label_schema": LABEL_SCHEMA,
        "label_support": dict(label_support),
        "notes": "Offline pure-Python classifier. Replace scoring backend later without changing output shape.",
    }
    write_json(config.models_dir / "legal_issue_classifier.json", model)
    return model


def _predict_from_features(row: dict, model: dict) -> dict:
    scores = dict(row["keyword_scores"])
    top_terms = set(row.get("top_terms", []))
    for label, meta in LABEL_SCHEMA.items():
        if label == "other":
            continue
        overlap = len(top_terms.intersection(meta["keywords"]))
        scores[label] = scores.get(label, 0.0) + overlap * 1.5
    if row.get("reference_count", 0) > 0:
        scores["statutory_interpretation"] = scores.get("statutory_interpretation", 0.0) + min(8.0, row["reference_count"] * 0.25)
    labels = _labels_from_scores(
        scores,
        threshold=model["threshold"],
        max_labels=model["max_labels_per_item"],
    )
    return {"id": row["id"], "item_type": row["item_type"], "labels": labels}


def predict_all(config: ClassificationConfig) -> dict:
    ensure_dir(config.predictions_dir)
    model = train_classifier(config)
    outputs = {}
    specs = [
        ("case", config.features_dir / "case_features.jsonl", config.predictions_dir / "case_predictions.jsonl"),
        ("statute", config.features_dir / "statute_features.jsonl", config.predictions_dir / "statute_predictions.jsonl"),
        ("query", config.features_dir / "query_features.jsonl", config.predictions_dir / "query_predictions.jsonl"),
    ]
    for name, features_path, output_path in specs:
        rows = [_predict_from_features(row, model) for row in read_jsonl(features_path)]
        write_jsonl(output_path, rows)
        outputs[name] = len(rows)
    write_json(config.predictions_dir / "prediction_summary.json", outputs)
    return outputs
