"""Pipeline 3 implementation: legal judgment summarization and explanation."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .io_utils import ensure_dir, read_json, read_jsonl, write_csv, write_json, write_jsonl
from .summarization_config import SummarizationConfig
from .summarization_core import extractive_summary, segment_document, select_section_sentences
from .text import snippet_for_query, token_count, top_terms
from .trec import read_run

ID_RE = re.compile(r"\b[CS]\d+\b")


def _label_map(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        return {}
    return {row["id"]: row.get("labels", []) for row in read_jsonl(path)}


def _labels_text(labels: list[dict]) -> list[str]:
    return [label["label"] for label in labels]


def segment_all(config: SummarizationConfig) -> dict:
    ensure_dir(config.processed_output_dir)
    cases = read_jsonl(config.processed_dir / "cases.jsonl")
    statutes = read_jsonl(config.processed_dir / "statutes.jsonl")
    case_maps = [segment_document(doc, "case") for doc in cases]
    statute_maps = [segment_document(doc, "statute", max_chunk_sentences=4) for doc in statutes]
    write_jsonl(config.processed_output_dir / "case_section_map.jsonl", case_maps)
    write_jsonl(config.processed_output_dir / "statute_section_map.jsonl", statute_maps)
    chunks = []
    for item in case_maps + statute_maps:
        chunks.extend(item["chunks"])
    write_jsonl(config.processed_output_dir / "chunks.jsonl", chunks)
    summary = {
        "cases_segmented": len(case_maps),
        "statutes_segmented": len(statute_maps),
        "chunks": len(chunks),
        "empty_cases": sum(1 for row in case_maps if row["sentence_count"] == 0),
        "empty_statutes": sum(1 for row in statute_maps if row["sentence_count"] == 0),
    }
    write_json(config.processed_output_dir / "segmentation_summary.json", summary)
    return summary


def summarize_cases(config: SummarizationConfig) -> dict:
    ensure_dir(config.summaries_dir)
    cases = read_jsonl(config.processed_dir / "cases.jsonl")
    case_labels = _label_map(config.classification_predictions_dir / "case_predictions.jsonl")
    rows = []
    for doc in cases:
        facts = select_section_sentences(doc["text"], "facts", limit=config.max_section_sentences)
        issues = select_section_sentences(doc["text"], "legal_issues", limit=config.max_section_sentences)
        reasoning = select_section_sentences(doc["text"], "reasoning", limit=config.max_section_sentences)
        holding = select_section_sentences(doc["text"], "holding", limit=config.max_section_sentences)
        flat = facts + issues + reasoning + holding
        if not flat:
            flat = extractive_summary(doc["text"], limit=config.max_case_summary_sentences)
        rows.append(
            {
                "doc_id": doc["doc_id"],
                "doc_type": "case",
                "labels": _labels_text(case_labels.get(doc["doc_id"], [])),
                "facts": facts,
                "legal_issues": issues,
                "reasoning": reasoning,
                "holding": holding,
                "important_terms": top_terms(doc["text"], limit=12),
                "summary_sentences": flat[: config.max_case_summary_sentences],
                "original_token_count": token_count(doc["text"]),
                "summary_token_count": token_count(" ".join(flat)),
                "extractive": True,
            }
        )
    write_jsonl(config.summaries_dir / "case_summaries.jsonl", rows)
    return {"case_summaries": len(rows)}


def summarize_statutes(config: SummarizationConfig) -> dict:
    ensure_dir(config.summaries_dir)
    statutes = read_jsonl(config.processed_dir / "statutes.jsonl")
    statute_labels = _label_map(config.classification_predictions_dir / "statute_predictions.jsonl")
    rows = []
    for doc in statutes:
        selected = extractive_summary(doc["text"], limit=3)
        title = doc.get("title", "")
        rows.append(
            {
                "doc_id": doc["doc_id"],
                "doc_type": "statute",
                "title": title,
                "labels": _labels_text(statute_labels.get(doc["doc_id"], [])),
                "core_rule": selected[0] if selected else doc["text"][:500],
                "summary_sentences": selected or [doc["text"][:500]],
                "important_terms": top_terms(doc["text"], limit=10),
                "original_token_count": token_count(doc["text"]),
                "summary_token_count": token_count(" ".join(selected)),
                "extractive": True,
            }
        )
    write_jsonl(config.summaries_dir / "statute_summaries.jsonl", rows)
    return {"statute_summaries": len(rows)}


def _run_by_query(path: Path) -> dict[str, list[dict]]:
    by_query: dict[str, list[dict]] = defaultdict(list)
    for row in read_run(path):
        by_query[row["query_id"]].append(row)
    for rows in by_query.values():
        rows.sort(key=lambda item: item["rank"])
    return dict(by_query)


def summarize_queries(config: SummarizationConfig) -> dict:
    ensure_dir(config.query_summaries_dir)
    queries = read_jsonl(config.processed_dir / "queries.jsonl")
    cases = {doc["doc_id"]: doc for doc in read_jsonl(config.processed_dir / "cases.jsonl")}
    statutes = {doc["doc_id"]: doc for doc in read_jsonl(config.processed_dir / "statutes.jsonl")}
    query_labels = _label_map(config.classification_predictions_dir / "query_predictions.jsonl")
    case_run = _run_by_query(config.runs_dir / f"{config.best_case_run}.trec")
    statute_run = _run_by_query(config.runs_dir / f"{config.best_statute_run}.trec")
    rows = []
    for query in queries:
        case_hits = case_run.get(query["query_id"], [])[: config.query_top_cases]
        statute_hits = statute_run.get(query["query_id"], [])[: config.query_top_statutes]
        case_summaries = []
        statute_summaries = []
        for hit in case_hits:
            doc = cases.get(hit["doc_id"])
            if not doc:
                continue
            sentences = extractive_summary(doc["text"], query=query["text"], limit=3)
            case_summaries.append(
                {
                    "doc_id": hit["doc_id"],
                    "rank": hit["rank"],
                    "score": hit["score"],
                    "summary_sentences": sentences,
                    "query_relevant_snippet": snippet_for_query(doc["text"], query["text"], max_chars=450),
                }
            )
        for hit in statute_hits:
            doc = statutes.get(hit["doc_id"])
            if not doc:
                continue
            sentences = extractive_summary(doc["text"], query=query["text"], limit=2)
            statute_summaries.append(
                {
                    "doc_id": hit["doc_id"],
                    "rank": hit["rank"],
                    "score": hit["score"],
                    "title": doc.get("title", ""),
                    "summary_sentences": sentences,
                }
            )
        case_ids = [item["doc_id"] for item in case_summaries]
        statute_ids = [item["doc_id"] for item in statute_summaries]
        labels = _labels_text(query_labels.get(query["query_id"], []))
        explanation = (
            f"Query {query['query_id']} is tagged as {', '.join(labels) if labels else 'unlabeled'}. "
            f"The query-focused summary is grounded in cases {', '.join(case_ids)} and statutes {', '.join(statute_ids)}."
        )
        rows.append(
            {
                "query_id": query["query_id"],
                "query": query["text"],
                "labels": labels,
                "case_run": config.best_case_run,
                "statute_run": config.best_statute_run,
                "case_summaries": case_summaries,
                "statute_summaries": statute_summaries,
                "combined_legal_explanation": explanation,
                "cited_case_ids": case_ids,
                "cited_statute_ids": statute_ids,
            }
        )
    write_jsonl(config.query_summaries_dir / "query_focused_summaries.jsonl", rows)
    return {"query_focused_summaries": len(rows)}


def evaluate_summaries(config: SummarizationConfig) -> dict:
    ensure_dir(config.evaluation_dir)
    errors = []
    case_docs = {row["doc_id"]: row for row in read_jsonl(config.processed_dir / "cases.jsonl")}
    statute_docs = {row["doc_id"]: row for row in read_jsonl(config.processed_dir / "statutes.jsonl")}
    cases = read_jsonl(config.summaries_dir / "case_summaries.jsonl")
    statutes = read_jsonl(config.summaries_dir / "statute_summaries.jsonl")
    queries = read_jsonl(config.query_summaries_dir / "query_focused_summaries.jsonl")
    valid_case_ids = set(case_docs)
    valid_statute_ids = set(statute_docs)
    retrieved_cases = _run_by_query(config.runs_dir / f"{config.best_case_run}.trec")
    retrieved_statutes = _run_by_query(config.runs_dir / f"{config.best_statute_run}.trec")

    total_summary_tokens = 0
    total_original_tokens = 0
    for row in cases:
        if row["doc_id"] not in valid_case_ids:
            errors.append({"type": "invalid_case_summary_id", "id": row["doc_id"]})
        if not row["summary_sentences"]:
            errors.append({"type": "empty_case_summary", "id": row["doc_id"]})
        source = case_docs.get(row["doc_id"], {}).get("text", "")
        for sentence in row["summary_sentences"]:
            if sentence and sentence not in source:
                errors.append({"type": "non_extractive_case_sentence", "id": row["doc_id"]})
                break
        total_summary_tokens += row["summary_token_count"]
        total_original_tokens += row["original_token_count"]
    for row in statutes:
        if row["doc_id"] not in valid_statute_ids:
            errors.append({"type": "invalid_statute_summary_id", "id": row["doc_id"]})
        if not row["summary_sentences"]:
            errors.append({"type": "empty_statute_summary", "id": row["doc_id"]})
        total_summary_tokens += row["summary_token_count"]
        total_original_tokens += row["original_token_count"]
    citation_checks = 0
    citation_valid = 0
    for row in queries:
        allowed_cases = {item["doc_id"] for item in retrieved_cases.get(row["query_id"], [])[: config.query_top_cases]}
        allowed_statutes = {item["doc_id"] for item in retrieved_statutes.get(row["query_id"], [])[: config.query_top_statutes]}
        for doc_id in row["cited_case_ids"]:
            citation_checks += 1
            if doc_id in valid_case_ids and doc_id in allowed_cases:
                citation_valid += 1
            else:
                errors.append({"type": "invalid_or_unretrieved_case_citation", "query_id": row["query_id"], "doc_id": doc_id})
        for doc_id in row["cited_statute_ids"]:
            citation_checks += 1
            if doc_id in valid_statute_ids and doc_id in allowed_statutes:
                citation_valid += 1
            else:
                errors.append({"type": "invalid_or_unretrieved_statute_citation", "query_id": row["query_id"], "doc_id": doc_id})
        cited_in_text = set(ID_RE.findall(row.get("combined_legal_explanation", "")))
        allowed_all = allowed_cases | allowed_statutes | set(row["cited_case_ids"]) | set(row["cited_statute_ids"])
        invented = sorted(cited_in_text - allowed_all)
        if invented:
            errors.append({"type": "hallucinated_id_in_query_summary", "query_id": row["query_id"], "ids": invented})

    report = {
        "case_summaries": len(cases),
        "statute_summaries": len(statutes),
        "query_focused_summaries": len(queries),
        "citation_checks": citation_checks,
        "citation_validity_rate": citation_valid / citation_checks if citation_checks else 1.0,
        "average_compression_ratio": total_summary_tokens / total_original_tokens if total_original_tokens else 0.0,
        "validation_error_count": len(errors),
        "passed": len(errors) == 0,
    }
    write_json(config.evaluation_dir / "summary_quality_report.json", report)
    write_json(config.evaluation_dir / "summary_validation_errors.json", {"errors": errors[:500], "total_errors": len(errors)})
    return report


def build_summarization_analytics(config: SummarizationConfig) -> dict:
    ensure_dir(config.analytics_dir)
    cases = read_jsonl(config.summaries_dir / "case_summaries.jsonl")
    statutes = read_jsonl(config.summaries_dir / "statute_summaries.jsonl")
    queries = read_jsonl(config.query_summaries_dir / "query_focused_summaries.jsonl")
    rows = []
    label_counts = Counter()
    review = []
    for collection, items in (("case", cases), ("statute", statutes)):
        for row in items:
            ratio = row["summary_token_count"] / row["original_token_count"] if row["original_token_count"] else 0.0
            rows.append(
                {
                    "collection": collection,
                    "id": row["doc_id"],
                    "original_tokens": row["original_token_count"],
                    "summary_tokens": row["summary_token_count"],
                    "compression_ratio": ratio,
                    "labels": ",".join(row.get("labels", [])),
                }
            )
            label_counts.update(row.get("labels", []))
            if row["summary_token_count"] == 0 or ratio > 0.7:
                review.append({"id": row["doc_id"], "collection": collection, "reason": "empty_or_low_compression"})
    write_csv(config.analytics_dir / "summary_statistics.csv", rows, ["collection", "id", "original_tokens", "summary_tokens", "compression_ratio", "labels"])
    write_csv(config.analytics_dir / "review_candidates.csv", review, ["id", "collection", "reason"])
    quality = read_json(config.evaluation_dir / "summary_quality_report.json")
    top_labels = label_counts.most_common(12)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pipeline 3 Legal Summarization</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    th {{ background: #f8fafc; }}
    .note {{ color: #475569; }}
  </style>
</head>
<body>
  <h1>Pipeline 3: Legal Judgment Summarization</h1>
  <p class="note">Extractive, citation-safe summaries built from AILA 2019 and Pipeline 1/2 outputs.</p>
  <table>
    <tr><th>Artifact</th><th>Count</th></tr>
    <tr><td>Case summaries</td><td>{len(cases)}</td></tr>
    <tr><td>Statute summaries</td><td>{len(statutes)}</td></tr>
    <tr><td>Query-focused summaries</td><td>{len(queries)}</td></tr>
    <tr><td>Citation validity</td><td>{quality['citation_validity_rate']:.4f}</td></tr>
    <tr><td>Average compression ratio</td><td>{quality['average_compression_ratio']:.4f}</td></tr>
    <tr><td>Validation errors</td><td>{quality['validation_error_count']}</td></tr>
  </table>
  <h2>Most Common Legal Issue Labels</h2>
  <table><tr><th>Label</th><th>Count</th></tr>{''.join(f'<tr><td>{label}</td><td>{count}</td></tr>' for label, count in top_labels)}</table>
  <h2>Example Query Summary</h2>
  <p><strong>{queries[0]['query_id'] if queries else 'n/a'}</strong>: {queries[0]['combined_legal_explanation'] if queries else ''}</p>
</body>
</html>
"""
    (config.analytics_dir / "pipeline3_summarization_report.html").write_text(html, encoding="utf-8")
    summary = {"summary_statistics_rows": len(rows), "review_candidates": len(review), "top_labels": top_labels}
    write_json(config.analytics_dir / "analytics_summary.json", summary)
    return summary
