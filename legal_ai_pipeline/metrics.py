"""trec_eval-style metrics in pure Python."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from .config import PipelineConfig
from .io_utils import read_jsonl, write_json
from .trec import read_run, validate_run


def qrels_to_dict(rows: list[dict]) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        qrels[row["query_id"]][row["doc_id"]] = int(row["relevance"])
    return dict(qrels)


def _dcg(relevances: list[int]) -> float:
    total = 0.0
    for idx, rel in enumerate(relevances, start=1):
        if rel <= 0:
            continue
        total += (2**rel - 1) / math.log2(idx + 1)
    return total


def evaluate_run(run_path: Path, qrels_path: Path, metrics_path: Path) -> dict:
    run_rows = read_run(run_path)
    validation = validate_run(run_path)
    qrels = qrels_to_dict(read_jsonl(qrels_path))
    by_query: dict[str, list[dict]] = defaultdict(list)
    for row in run_rows:
        by_query[row["query_id"]].append(row)
    for rows in by_query.values():
        rows.sort(key=lambda r: r["rank"])

    query_metrics = {}
    cutoffs_p = [5, 10]
    cutoffs_r = [10, 50, 100]
    cutoffs_n = [10, 100]
    for qid in sorted(qrels):
        relevant = {doc_id for doc_id, rel in qrels[qid].items() if rel > 0}
        ranked = by_query.get(qid, [])
        ranked_ids = [r["doc_id"] for r in ranked]
        num_rel = len(relevant)
        hits = 0
        precision_sum = 0.0
        reciprocal_rank = 0.0
        rel_sequence = []
        for rank, doc_id in enumerate(ranked_ids, start=1):
            rel = 1 if doc_id in relevant else 0
            rel_sequence.append(rel)
            if rel:
                hits += 1
                precision_sum += hits / rank
                if reciprocal_rank == 0.0:
                    reciprocal_rank = 1.0 / rank
        per = {
            "num_relevant": num_rel,
            "retrieved": len(ranked_ids),
            "ap": precision_sum / num_rel if num_rel else 0.0,
            "rr": reciprocal_rank,
        }
        for k in cutoffs_p:
            top = ranked_ids[:k]
            per[f"p@{k}"] = sum(1 for doc in top if doc in relevant) / k
        for k in cutoffs_r:
            top = ranked_ids[:k]
            per[f"recall@{k}"] = sum(1 for doc in top if doc in relevant) / num_rel if num_rel else 0.0
        for k in cutoffs_n:
            actual = _dcg(rel_sequence[:k])
            ideal_rels = [1] * min(num_rel, k)
            ideal = _dcg(ideal_rels)
            per[f"ndcg@{k}"] = actual / ideal if ideal else 0.0
        query_metrics[qid] = per

    metric_keys = ["ap", "rr", "p@5", "p@10", "recall@10", "recall@50", "recall@100", "ndcg@10", "ndcg@100"]
    overall = {}
    for key in metric_keys:
        values = [query_metrics[qid][key] for qid in query_metrics]
        overall["map" if key == "ap" else "mrr" if key == "rr" else key] = sum(values) / len(values) if values else 0.0

    output = {
        "run_file": str(run_path),
        "qrels_file": str(qrels_path),
        "validation": validation,
        "overall": overall,
        "per_query": query_metrics,
    }
    write_json(metrics_path, output)
    return output


def evaluate_standard_runs(config: PipelineConfig) -> list[dict]:
    specs = [
        ("bm25_cases", config.processed_dir / "qrels_cases.jsonl"),
        ("bm25_statutes", config.processed_dir / "qrels_statutes.jsonl"),
        ("doc2query_bm25_cases", config.processed_dir / "qrels_cases.jsonl"),
        ("doc2query_bm25_statutes", config.processed_dir / "qrels_statutes.jsonl"),
        ("colbert_cases", config.processed_dir / "qrels_cases.jsonl"),
        ("colbert_statutes", config.processed_dir / "qrels_statutes.jsonl"),
        ("bm25_colbert_cases", config.processed_dir / "qrels_cases.jsonl"),
        ("bm25_colbert_statutes", config.processed_dir / "qrels_statutes.jsonl"),
        ("doc2query_colbert_cases", config.processed_dir / "qrels_cases.jsonl"),
        ("doc2query_colbert_statutes", config.processed_dir / "qrels_statutes.jsonl"),
    ]
    results = []
    for run_name, qrels_path in specs:
        run_path = config.runs_dir / f"{run_name}.trec"
        if run_path.exists():
            results.append(evaluate_run(run_path, qrels_path, config.metrics_dir / f"{run_name}.json"))
    return results
