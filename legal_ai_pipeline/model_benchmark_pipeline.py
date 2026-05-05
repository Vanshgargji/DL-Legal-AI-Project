"""Final pipeline: model benchmarking and upgrade harness.

The current workspace has no ML libraries installed, so every stage has a
deterministic fallback and records whether a real neural backend was available.
"""

from __future__ import annotations

import importlib.util
import json
import math
import platform
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from .classification_features import legal_keyword_scores
from .classification_labels import _labels_from_scores
from .classification_schema import LABEL_SCHEMA
from .config import PipelineConfig
from .io_utils import ensure_dir, read_json, read_jsonl, write_csv, write_json, write_jsonl
from .metrics import evaluate_run
from .model_benchmark_config import ModelBenchmarkConfig
from .text import normalize_text, snippet_for_query, tokenize, top_terms
from .trec import read_run, write_run


MODEL_REGISTRY = {
    "dense_retrieval": [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ],
    "legal_classification": [
        "law-ai/InLegalBERT",
        "nlpaueb/legal-bert-base-uncased",
        "bert-base-uncased",
    ],
    "reranking": [
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "law-ai/InLegalBERT",
    ],
    "late_interaction": ["colbert-ir/colbertv2.0"],
    "summarization": [
        "allenai/led-base-16384",
        "pszemraj/long-t5-tglobal-base-16384-book-summary",
        "facebook/bart-large-cnn",
    ],
}

REQUIRED_PACKAGES = {
    "torch": "Core tensor/model runtime",
    "transformers": "Legal-BERT/InLegalBERT/LED model loading",
    "sentence_transformers": "Dense retrieval embeddings and cross-encoder utilities",
    "faiss": "Fast vector indexing",
    "sklearn": "Model training and classification metrics",
    "numpy": "Vector math backend",
    "colbert": "Real ColBERTv2 late-interaction retrieval",
}


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def environment_check(config: ModelBenchmarkConfig) -> dict:
    ensure_dir(config.output_dir)
    ensure_dir(config.models_dir)
    packages = {name: {"installed": _has_module(name), "purpose": purpose} for name, purpose in REQUIRED_PACKAGES.items()}
    torch_info = {"available": False, "cuda_available": False, "device": "cpu"}
    if packages["torch"]["installed"]:
        try:
            import torch  # type: ignore

            torch_info = {
                "available": True,
                "cuda_available": bool(torch.cuda.is_available()),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
        except Exception as exc:  # pragma: no cover - defensive runtime report
            torch_info = {"available": False, "cuda_available": False, "device": "cpu", "error": str(exc)}
    report = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "disk_free_gb": round(shutil.disk_usage(config.project_root).free / 1024**3, 2),
        "packages": packages,
        "torch": torch_info,
        "can_run_full_transformers": packages["torch"]["installed"] and packages["transformers"]["installed"],
        "can_run_dense_retrieval": packages["sentence_transformers"]["installed"],
        "can_run_faiss": packages["faiss"]["installed"],
        "can_run_colbert": packages["colbert"]["installed"],
        "fallback_mode": not (packages["torch"]["installed"] and packages["transformers"]["installed"]),
    }
    write_json(config.output_dir / "environment_report.json", report)
    write_json(config.models_dir / "model_registry.json", {"recommended_models": MODEL_REGISTRY, "environment": report})
    requirements = "\n".join(
        [
            "torch",
            "transformers",
            "sentence-transformers",
            "faiss-cpu",
            "scikit-learn",
            "numpy",
            "pandas",
            "colbert-ai",
        ]
    )
    (config.output_dir / "requirements-models.txt").write_text(requirements + "\n", encoding="utf-8")
    install_notes = (
        "# Model Pipeline Install Notes\n\n"
        "Install these only when you are ready to run real transformer models.\n\n"
        "```powershell\n"
        "python -m pip install -r model_benchmark_outputs/requirements-models.txt\n"
        "```\n\n"
        "For GPU PyTorch, install the CUDA-specific wheel from the official PyTorch selector instead of the default CPU wheel.\n"
    )
    (config.output_dir / "MODEL_INSTALL_NOTES.md").write_text(install_notes, encoding="utf-8")
    return report


def _idf_vectors(docs: list[dict]) -> tuple[list[Counter], dict[str, float]]:
    doc_terms = []
    df = Counter()
    for doc in docs:
        terms = Counter(tokenize(doc["text"], remove_stopwords=True))
        doc_terms.append(terms)
        df.update(terms.keys())
    n = max(1, len(docs))
    idf = {term: math.log(1 + n / (1 + freq)) for term, freq in df.items()}
    return doc_terms, idf


def _cosine_sparse(query_counts: Counter, doc_counts: Counter, idf: dict[str, float]) -> float:
    numerator = 0.0
    q_norm = 0.0
    d_norm = 0.0
    for term, q_count in query_counts.items():
        q_weight = q_count * idf.get(term, 1.0)
        q_norm += q_weight * q_weight
        if term in doc_counts:
            numerator += q_weight * doc_counts[term] * idf.get(term, 1.0)
    for term, d_count in doc_counts.items():
        d_weight = d_count * idf.get(term, 1.0)
        d_norm += d_weight * d_weight
    return numerator / math.sqrt(q_norm * d_norm) if q_norm and d_norm else 0.0


def _dense_fallback_run(docs: list[dict], queries: list[dict], top_k: int, run_id: str, run_path: Path) -> dict:
    doc_counts, idf = _idf_vectors(docs)
    rows = []
    for query in queries:
        q_counts = Counter(tokenize(query["text"], remove_stopwords=True))
        scored = [(_cosine_sparse(q_counts, doc_counts[idx], idf), docs[idx]["doc_id"]) for idx in range(len(docs))]
        scored.sort(key=lambda item: (-item[0], item[1]))
        for rank, (score, doc_id) in enumerate(scored[:top_k], start=1):
            rows.append({"query_id": query["query_id"], "doc_id": doc_id, "rank": rank, "score": score, "run_id": run_id})
    write_run(run_path, rows)
    return {"run_id": run_id, "run_path": str(run_path), "rows": len(rows), "backend": "tfidf_dense_fallback"}


def run_dense_retrieval(config: ModelBenchmarkConfig) -> dict:
    ensure_dir(config.runs_dir)
    ensure_dir(config.embeddings_dir)
    env = read_json(config.output_dir / "environment_report.json") if (config.output_dir / "environment_report.json").exists() else environment_check(config)
    cases = read_jsonl(config.processed_dir / "cases.jsonl")
    statutes = read_jsonl(config.processed_dir / "statutes.jsonl")
    queries = read_jsonl(config.processed_dir / "queries.jsonl")
    if env.get("can_run_dense_retrieval"):
        # Real model implementation hook intentionally isolated for dependency-backed upgrade.
        backend = "sentence_transformers_available_not_executed_in_safe_default"
    else:
        backend = "tfidf_dense_fallback"
    outputs = {
        "backend": backend,
        "cases": _dense_fallback_run(cases, queries, config.top_k_cases, "dense_retrieval_cases", config.runs_dir / "dense_retrieval_cases.trec"),
        "statutes": _dense_fallback_run(statutes, queries, config.top_k_statutes, "dense_retrieval_statutes", config.runs_dir / "dense_retrieval_statutes.trec"),
    }
    write_json(config.embeddings_dir / "dense_retrieval_manifest.json", outputs)
    return outputs


def run_cross_encoder_rerank(config: ModelBenchmarkConfig) -> dict:
    ensure_dir(config.runs_dir)
    cases = {doc["doc_id"]: doc for doc in read_jsonl(config.processed_dir / "cases.jsonl")}
    statutes = {doc["doc_id"]: doc for doc in read_jsonl(config.processed_dir / "statutes.jsonl")}
    queries = {q["query_id"]: q for q in read_jsonl(config.processed_dir / "queries.jsonl")}

    def rerank(candidate_path: Path, docs: dict[str, dict], top_k: int, run_id: str, output_path: Path) -> dict:
        by_query: dict[str, list[dict]] = defaultdict(list)
        for row in read_run(candidate_path):
            by_query[row["query_id"]].append(row)
        rows = []
        for qid, candidates in by_query.items():
            query = queries[qid]["text"]
            q_terms = set(tokenize(query, remove_stopwords=True))
            scored = []
            for cand in sorted(candidates, key=lambda item: item["rank"])[: config.rerank_depth]:
                doc = docs.get(cand["doc_id"])
                if not doc:
                    continue
                snippet = normalize_text(doc["text"][:12000])
                d_terms = set(tokenize(snippet, remove_stopwords=True))
                overlap = len(q_terms & d_terms) / max(1, len(q_terms))
                phrase_bonus = sum(1 for term in q_terms if term in snippet.lower()) / max(1, len(q_terms))
                score = overlap * 0.7 + phrase_bonus * 0.2 + (1 / cand["rank"]) * 0.1
                scored.append((score, cand["doc_id"]))
            scored.sort(key=lambda item: (-item[0], item[1]))
            for rank, (score, doc_id) in enumerate(scored[:top_k], start=1):
                rows.append({"query_id": qid, "doc_id": doc_id, "rank": rank, "score": score, "run_id": run_id})
        write_run(output_path, rows)
        return {"run_id": run_id, "run_path": str(output_path), "rows": len(rows), "backend": "cross_encoder_overlap_fallback"}

    return {
        "cases": rerank(
            config.baseline_runs_dir / "doc2query_bm25_cases.trec",
            cases,
            config.rerank_depth,
            "cross_encoder_rerank_cases",
            config.runs_dir / "cross_encoder_rerank_cases.trec",
        ),
        "statutes": rerank(
            config.baseline_runs_dir / "bm25_statutes.trec",
            statutes,
            config.top_k_statutes,
            "cross_encoder_rerank_statutes",
            config.runs_dir / "cross_encoder_rerank_statutes.trec",
        ),
    }


def run_transformer_classification(config: ModelBenchmarkConfig) -> dict:
    ensure_dir(config.classification_dir)
    outputs = []
    for source_name, target_name in (
        ("case_predictions.jsonl", "transformer_case_predictions.jsonl"),
        ("statute_predictions.jsonl", "transformer_statute_predictions.jsonl"),
        ("query_predictions.jsonl", "transformer_query_predictions.jsonl"),
    ):
        rows = read_jsonl(config.classification_predictions_dir / source_name)
        converted = []
        for row in rows:
            labels = []
            for label in row.get("labels", []):
                copy = dict(label)
                copy["source"] = "pipeline2_silver_fallback_for_transformer_classifier"
                labels.append(copy)
            converted.append({"id": row["id"], "item_type": row["item_type"], "labels": labels})
        write_jsonl(config.classification_dir / target_name, converted)
        outputs.append({"file": target_name, "rows": len(converted)})
    merged = []
    for target in ("transformer_case_predictions.jsonl", "transformer_statute_predictions.jsonl", "transformer_query_predictions.jsonl"):
        merged.extend(read_jsonl(config.classification_dir / target))
    write_jsonl(config.classification_dir / "transformer_classification_predictions.jsonl", merged)
    metrics = {
        "backend": "pipeline2_silver_fallback",
        "files": outputs,
        "note": "Install torch/transformers to train InLegalBERT or Legal-BERT classifier.",
    }
    write_json(config.classification_dir / "classification_training_metrics.json", metrics)
    return metrics


def run_colbert_manifest(config: ModelBenchmarkConfig) -> dict:
    ensure_dir(config.runs_dir)
    env = read_json(config.output_dir / "environment_report.json") if (config.output_dir / "environment_report.json").exists() else environment_check(config)
    outputs = {"backend": "colbert_unavailable_manifest_only", "can_run_colbert": env.get("can_run_colbert", False)}
    for src, dst in (
        ("doc2query_colbert_cases.trec", "real_colbert_cases.trec"),
        ("bm25_colbert_statutes.trec", "real_colbert_statutes.trec"),
    ):
        src_path = config.baseline_runs_dir / src
        dst_path = config.runs_dir / dst
        if src_path.exists():
            dst_path.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")
            outputs[dst] = {"rows": sum(1 for _ in dst_path.open("r", encoding="utf-8")), "source": src, "backend": "pipeline1_colbert_fallback_copy"}
    write_json(config.models_dir / "real_colbert_manifest.json", outputs)
    return outputs


def run_transformer_summarization(config: ModelBenchmarkConfig) -> dict:
    ensure_dir(config.summarization_output_dir)
    source_cases = config.summarization_dir / "summaries" / "case_summaries.jsonl"
    source_statutes = config.summarization_dir / "summaries" / "statute_summaries.jsonl"
    case_rows = read_jsonl(source_cases)
    statute_rows = read_jsonl(source_statutes)
    transformed = []
    for row in case_rows:
        transformed.append(
            {
                "doc_id": row["doc_id"],
                "doc_type": "case",
                "backend": "pipeline3_extractive_fallback_for_transformer_summarizer",
                "summary": " ".join(row.get("summary_sentences", [])),
                "labels": row.get("labels", []),
            }
        )
    for row in statute_rows:
        transformed.append(
            {
                "doc_id": row["doc_id"],
                "doc_type": "statute",
                "backend": "pipeline3_extractive_fallback_for_transformer_summarizer",
                "summary": " ".join(row.get("summary_sentences", [])),
                "labels": row.get("labels", []),
            }
        )
    write_jsonl(config.summarization_output_dir / "transformer_summaries.jsonl", transformed)
    metrics = {"backend": "pipeline3_extractive_fallback", "summaries": len(transformed)}
    write_json(config.summarization_output_dir / "summarization_quality_metrics.json", metrics)
    return metrics


def run_model_rag(config: ModelBenchmarkConfig) -> dict:
    ensure_dir(config.rag_output_dir)
    source_answers = config.rag_dir / "answers.jsonl"
    rows = read_jsonl(source_answers)
    converted = []
    unsupported = []
    for row in rows:
        cited = row.get("citations", {})
        answer = row.get("answer", "")
        converted.append(
            {
                "query_id": row["query_id"],
                "backend": "pipeline1_rag_fallback_for_model_rag",
                "answer": answer,
                "citations": cited,
                "audit": row.get("audit", {}),
            }
        )
        if not row.get("audit", {}).get("all_citations_supported", False):
            unsupported.append(row["query_id"])
    write_jsonl(config.rag_output_dir / "model_rag_answers.jsonl", converted)
    audit = {"answers": len(converted), "unsupported_citation_queries": unsupported, "citation_validity_rate": 1.0 if not unsupported else 0.0}
    write_json(config.rag_output_dir / "model_rag_citation_audit.json", audit)
    write_json(config.rag_output_dir / "model_rag_comparison.json", {"backend": "pipeline1_fallback", "answers": len(converted)})
    return audit


def evaluate_model_runs(config: ModelBenchmarkConfig) -> dict:
    ensure_dir(config.metrics_dir)
    specs = [
        ("dense_retrieval_cases", config.processed_dir / "qrels_cases.jsonl"),
        ("dense_retrieval_statutes", config.processed_dir / "qrels_statutes.jsonl"),
        ("cross_encoder_rerank_cases", config.processed_dir / "qrels_cases.jsonl"),
        ("cross_encoder_rerank_statutes", config.processed_dir / "qrels_statutes.jsonl"),
        ("real_colbert_cases", config.processed_dir / "qrels_cases.jsonl"),
        ("real_colbert_statutes", config.processed_dir / "qrels_statutes.jsonl"),
    ]
    outputs = []
    for name, qrels in specs:
        run_path = config.runs_dir / f"{name}.trec"
        if run_path.exists():
            outputs.append(evaluate_run(run_path, qrels, config.metrics_dir / f"{name}.json"))
    summary = []
    for result in outputs:
        summary.append(
            {
                "run": Path(result["run_file"]).stem,
                "map": result["overall"]["map"],
                "ndcg@10": result["overall"]["ndcg@10"],
                "recall@100": result["overall"]["recall@100"],
            }
        )
    write_json(config.metrics_dir / "model_benchmark_metrics.json", {"runs": summary})
    return {"evaluated_runs": len(outputs), "runs": summary}


def build_model_report(config: ModelBenchmarkConfig) -> dict:
    ensure_dir(config.analytics_dir)
    env = read_json(config.output_dir / "environment_report.json")
    model_metrics = read_json(config.metrics_dir / "model_benchmark_metrics.json") if (config.metrics_dir / "model_benchmark_metrics.json").exists() else {"runs": []}
    baseline_rows = []
    for path in sorted(config.baseline_metrics_dir.glob("*.json")):
        data = read_json(path)
        baseline_rows.append(
            {
                "run": path.stem,
                "source": "pipeline1",
                "map": data["overall"]["map"],
                "ndcg@10": data["overall"]["ndcg@10"],
                "recall@100": data["overall"]["recall@100"],
            }
        )
    model_rows = [
        {"run": row["run"], "source": "model_benchmark", "map": row["map"], "ndcg@10": row["ndcg@10"], "recall@100": row["recall@100"]}
        for row in model_metrics.get("runs", [])
    ]
    all_rows = baseline_rows + model_rows
    write_csv(config.analytics_dir / "retrieval_model_comparison.csv", all_rows, ["run", "source", "map", "ndcg@10", "recall@100"])
    best = max(all_rows, key=lambda row: row["ndcg@10"], default=None)
    recommendations = {
        "best_retrieval_by_ndcg@10": best,
        "environment_fallback_mode": env.get("fallback_mode", True),
        "next_step": "Install ML dependencies and rerun this pipeline to execute real transformer models." if env.get("fallback_mode") else "Run full transformer training and compare against fallback baselines.",
    }
    write_json(config.analytics_dir / "best_model_recommendations.json", recommendations)
    html_rows = "".join(
        f"<tr><td>{row['run']}</td><td>{row['source']}</td><td>{row['map']:.4f}</td><td>{row['ndcg@10']:.4f}</td><td>{row['recall@100']:.4f}</td></tr>"
        for row in sorted(all_rows, key=lambda row: (-row["ndcg@10"], row["run"]))
    )
    missing = [name for name, info in env["packages"].items() if not info["installed"]]
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Final Model Benchmark Pipeline</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    th {{ background: #f8fafc; }}
    .note {{ color: #475569; }}
  </style>
</head>
<body>
  <h1>Final Model Benchmark Pipeline</h1>
  <p class="note">Fallback mode: {env.get('fallback_mode')}. Missing model packages: {', '.join(missing) if missing else 'none'}.</p>
  <h2>Retrieval Comparison</h2>
  <table><tr><th>Run</th><th>Source</th><th>MAP</th><th>nDCG@10</th><th>Recall@100</th></tr>{html_rows}</table>
  <h2>Recommendation</h2>
  <p>{recommendations['next_step']}</p>
</body>
</html>
"""
    (config.analytics_dir / "model_comparison_report.html").write_text(html, encoding="utf-8")
    write_json(config.analytics_dir / "analytics_summary.json", {"rows": len(all_rows), "missing_packages": missing, "best": best})
    return recommendations


def run_all(config: ModelBenchmarkConfig) -> dict:
    for directory in (
        config.models_dir,
        config.embeddings_dir,
        config.runs_dir,
        config.classification_dir,
        config.summarization_output_dir,
        config.rag_output_dir,
        config.metrics_dir,
        config.analytics_dir,
    ):
        ensure_dir(directory)
    return {
        "environment": environment_check(config),
        "dense_retrieval": run_dense_retrieval(config),
        "cross_encoder_rerank": run_cross_encoder_rerank(config),
        "classification": run_transformer_classification(config),
        "colbert": run_colbert_manifest(config),
        "summarization": run_transformer_summarization(config),
        "rag": run_model_rag(config),
        "evaluation": evaluate_model_runs(config),
        "analytics": build_model_report(config),
    }
