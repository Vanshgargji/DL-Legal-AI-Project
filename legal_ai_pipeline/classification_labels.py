"""Weak supervision label generation for Pipeline 2."""

from __future__ import annotations

from collections import defaultdict

from .classification_config import ClassificationConfig
from .classification_features import extract_features, legal_keyword_scores
from .classification_schema import LABEL_SCHEMA
from .io_utils import ensure_dir, read_jsonl, write_json, write_jsonl


def _labels_from_scores(scores: dict[str, float], *, threshold: float, max_labels: int) -> list[dict]:
    if not scores:
        return [{"label": "other", "confidence": 0.35, "source": "fallback"}]
    max_score = max(scores.values()) if scores else 0.0
    if max_score <= 0:
        return [{"label": "other", "confidence": 0.35, "source": "fallback"}]
    labels = []
    for label, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
        confidence = min(0.97, score / (max_score + 2.0) + min(0.2, score / 60.0))
        if confidence >= threshold:
            labels.append({"label": label, "confidence": round(confidence, 4), "source": "keyword_weak_supervision"})
        if len(labels) >= max_labels:
            break
    return labels or [{"label": "other", "confidence": 0.35, "source": "fallback"}]


def _qrels_positive(path) -> dict[str, set[str]]:
    rows = read_jsonl(path)
    out: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["relevance"] > 0:
            out[row["query_id"]].add(row["doc_id"])
    return dict(out)


def generate_weak_labels(config: ClassificationConfig) -> dict:
    ensure_dir(config.labels_dir)
    cases = read_jsonl(config.processed_dir / "cases.jsonl")
    statutes = read_jsonl(config.processed_dir / "statutes.jsonl")
    queries = read_jsonl(config.processed_dir / "queries.jsonl")
    query_to_cases = _qrels_positive(config.processed_dir / "qrels_cases.jsonl")
    query_to_statutes = _qrels_positive(config.processed_dir / "qrels_statutes.jsonl")

    case_labels = []
    statute_labels = []
    query_labels = []
    label_by_doc = {}

    for doc in cases:
        labels = _labels_from_scores(legal_keyword_scores(doc["text"]), threshold=config.confidence_threshold, max_labels=config.max_labels_per_item)
        row = {"id": doc["doc_id"], "item_type": "case", "labels": labels}
        case_labels.append(row)
        label_by_doc[doc["doc_id"]] = labels

    for doc in statutes:
        labels = _labels_from_scores(
            legal_keyword_scores((doc.get("title", "") + " " + doc["text"]).strip()),
            threshold=config.confidence_threshold,
            max_labels=config.max_labels_per_item,
        )
        row = {"id": doc["doc_id"], "item_type": "statute", "labels": labels}
        statute_labels.append(row)
        label_by_doc[doc["doc_id"]] = labels

    for query in queries:
        score = legal_keyword_scores(query["text"])
        for doc_id in query_to_cases.get(query["query_id"], set()) | query_to_statutes.get(query["query_id"], set()):
            for label in label_by_doc.get(doc_id, []):
                score[label["label"]] = score.get(label["label"], 0.0) + label["confidence"] * 2.0
        labels = _labels_from_scores(score, threshold=config.confidence_threshold, max_labels=config.max_labels_per_item)
        query_labels.append({"id": query["query_id"], "item_type": "query", "labels": labels})

    all_labels = case_labels + statute_labels + query_labels
    write_jsonl(config.labels_dir / "weak_case_labels.jsonl", case_labels)
    write_jsonl(config.labels_dir / "weak_statute_labels.jsonl", statute_labels)
    write_jsonl(config.labels_dir / "weak_query_labels.jsonl", query_labels)
    write_jsonl(config.labels_dir / "train_dataset.jsonl", all_labels)

    coverage = {
        "cases_labeled": sum(1 for r in case_labels if r["labels"]),
        "statutes_labeled": sum(1 for r in statute_labels if r["labels"]),
        "queries_labeled": sum(1 for r in query_labels if r["labels"]),
        "cases_total": len(case_labels),
        "statutes_total": len(statute_labels),
        "queries_total": len(query_labels),
        "schema_labels": sorted(LABEL_SCHEMA),
    }
    write_json(config.labels_dir / "weak_label_summary.json", coverage)
    return coverage
