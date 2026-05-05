"""Citation-grounded legal RAG outputs built from saved retrieval runs."""

from __future__ import annotations

import re
from collections import defaultdict

from .config import PipelineConfig
from .io_utils import read_jsonl, read_json, write_json, write_jsonl
from .text import snippet_for_query, top_terms
from .trec import read_run


CID_RE = re.compile(r"\b[CS]\d+\b")


def _best_metric_run(config: PipelineConfig, collection: str) -> str:
    suffix = "_cases" if collection == "cases" else "_statutes"
    best_name = ""
    best_score = -1.0
    for path in config.metrics_dir.glob(f"*{suffix}.json"):
        data = read_json(path)
        score = data.get("overall", {}).get("ndcg@10", 0.0)
        if score > best_score:
            best_score = score
            best_name = path.stem
    return best_name


def _run_by_query(run_path) -> dict[str, list[dict]]:
    rows = read_run(run_path)
    by_query = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)
    for ranked in by_query.values():
        ranked.sort(key=lambda r: r["rank"])
    return dict(by_query)


def _answer_for_query(query: dict, case_hits: list[dict], statute_hits: list[dict], case_docs: dict, statute_docs: dict) -> dict:
    case_parts = []
    statute_parts = []
    for hit in case_hits:
        doc = case_docs.get(hit["doc_id"])
        if doc:
            case_parts.append(
                {
                    "doc_id": hit["doc_id"],
                    "rank": hit["rank"],
                    "score": hit["score"],
                    "snippet": snippet_for_query(doc["text"], query["text"]),
                }
            )
    for hit in statute_hits:
        doc = statute_docs.get(hit["doc_id"])
        if doc:
            statute_parts.append(
                {
                    "doc_id": hit["doc_id"],
                    "rank": hit["rank"],
                    "score": hit["score"],
                    "title": doc.get("title", ""),
                    "snippet": snippet_for_query(doc["text"], query["text"], max_chars=500),
                }
            )

    case_ids = [item["doc_id"] for item in case_parts]
    statute_ids = [item["doc_id"] for item in statute_parts]
    issue_terms = ", ".join(top_terms(query["text"], limit=8))
    evidence_sentence = ""
    if case_ids and statute_ids:
        evidence_sentence = f"The strongest retrieved authorities are prior cases {', '.join(case_ids)} and statutes {', '.join(statute_ids)}."
    elif case_ids:
        evidence_sentence = f"The strongest retrieved authorities are prior cases {', '.join(case_ids)}; no high-confidence statute citation was retrieved."
    elif statute_ids:
        evidence_sentence = f"The strongest retrieved authorities are statutes {', '.join(statute_ids)}; no high-confidence prior case citation was retrieved."
    else:
        evidence_sentence = "The retrieval layer did not return enough evidence for a grounded legal answer."

    answer = (
        f"For query {query['query_id']}, the apparent legal issues include {issue_terms}. "
        f"{evidence_sentence} "
        "This draft answer is retrieval-grounded: use the cited IDs to inspect the source text before relying on the conclusion."
    )
    cited_ids = sorted(set(CID_RE.findall(answer)))
    context_ids = set(case_ids + statute_ids)
    unsupported = [doc_id for doc_id in cited_ids if doc_id not in context_ids]
    weak = len(case_ids) < 2 or len(statute_ids) < 1
    return {
        "query_id": query["query_id"],
        "query": query["text"],
        "answer": answer,
        "citations": {"cases": case_ids, "statutes": statute_ids},
        "evidence": {"cases": case_parts, "statutes": statute_parts},
        "audit": {
            "cited_ids": cited_ids,
            "unsupported_citations": unsupported,
            "all_citations_supported": not unsupported,
            "weak_retrieval_coverage": weak,
        },
    }


def run_rag(config: PipelineConfig) -> dict:
    queries = read_jsonl(config.processed_dir / "queries.jsonl")
    cases = {doc["doc_id"]: doc for doc in read_jsonl(config.processed_dir / "cases.jsonl")}
    statutes = {doc["doc_id"]: doc for doc in read_jsonl(config.processed_dir / "statutes.jsonl")}
    best_cases = _best_metric_run(config, "cases")
    best_statutes = _best_metric_run(config, "statutes")
    case_runs = _run_by_query(config.runs_dir / f"{best_cases}.trec")
    statute_runs = _run_by_query(config.runs_dir / f"{best_statutes}.trec")

    answers = []
    traces = []
    audits = []
    for query in queries:
        case_hits = case_runs.get(query["query_id"], [])[: config.rag_top_cases]
        statute_hits = statute_runs.get(query["query_id"], [])[: config.rag_top_statutes]
        answer = _answer_for_query(query, case_hits, statute_hits, cases, statutes)
        answers.append(answer)
        traces.append(
            {
                "query_id": query["query_id"],
                "case_run": best_cases,
                "statute_run": best_statutes,
                "case_hits": case_hits,
                "statute_hits": statute_hits,
            }
        )
        audits.append({"query_id": query["query_id"], **answer["audit"]})

    write_jsonl(config.rag_dir / "answers.jsonl", answers)
    write_jsonl(config.rag_dir / "retrieval_traces.jsonl", traces)
    write_json(config.rag_dir / "citation_audit.json", {"items": audits})
    summary = {
        "answers": len(answers),
        "best_case_run": best_cases,
        "best_statute_run": best_statutes,
        "unsupported_citation_queries": [a["query_id"] for a in audits if a["unsupported_citations"]],
        "weak_retrieval_queries": [a["query_id"] for a in audits if a["weak_retrieval_coverage"]],
    }
    write_json(config.rag_dir / "rag_summary.json", summary)
    return summary
