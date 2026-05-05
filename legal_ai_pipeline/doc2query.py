"""Offline Doc2Query-style document expansion.

This module keeps the Doc2Query interface but uses a deterministic legal-keyphrase
fallback when no local generation model is installed.
"""

from __future__ import annotations

from .bm25 import build_bm25_run
from .config import PipelineConfig
from .io_utils import read_jsonl, write_json, write_jsonl
from .text import top_terms


def generate_expansions(doc: dict, max_expansions: int = 5) -> list[str]:
    terms = top_terms((doc.get("title", "") + " " + doc["text"])[:20000], limit=14)
    if not terms:
        return []
    expansions = []
    lead = terms[:4]
    expansions.append("legal issue involving " + " ".join(lead))
    if any(t in terms for t in ("section", "article", "act", "constitution")):
        expansions.append("statutory interpretation of " + " ".join(terms[:5]))
    if any(t in terms for t in ("appeal", "appellant", "respondent", "petition", "writ")):
        expansions.append("court proceeding concerning " + " ".join(terms[:5]))
    if any(t in terms for t in ("property", "contract", "liability", "relief", "claim")):
        expansions.append("rights and relief for " + " ".join(terms[:5]))
    expansions.append("relevant precedent on " + " ".join(terms[:6]))
    deduped = []
    seen = set()
    for expansion in expansions:
        if expansion not in seen:
            deduped.append(expansion)
            seen.add(expansion)
    return deduped[:max_expansions]


def expand_collection(docs: list[dict], cache_path) -> list[dict]:
    cached = []
    for doc in docs:
        expansions = generate_expansions(doc)
        cached.append({"doc_id": doc["doc_id"], "expansions": expansions, "generator": "offline_legal_keyphrase_fallback"})
    write_jsonl(cache_path, cached)
    expansions_by_id = {row["doc_id"]: row["expansions"] for row in cached}
    expanded = []
    for doc in docs:
        copy = dict(doc)
        copy["expansions"] = expansions_by_id[doc["doc_id"]]
        copy["text"] = doc["text"] + "\n\n" + "\n".join(copy["expansions"])
        expanded.append(copy)
    return expanded


def run_doc2query(config: PipelineConfig) -> list[dict]:
    cases = read_jsonl(config.processed_dir / "cases.jsonl")
    statutes = read_jsonl(config.processed_dir / "statutes.jsonl")
    queries = read_jsonl(config.processed_dir / "queries.jsonl")
    cases_expanded = expand_collection(cases, config.processed_dir / "doc2query_cases_expansions.jsonl")
    statutes_expanded = expand_collection(statutes, config.processed_dir / "doc2query_statutes_expansions.jsonl")
    write_jsonl(config.processed_dir / "cases_doc2query.jsonl", cases_expanded)
    write_jsonl(config.processed_dir / "statutes_doc2query.jsonl", statutes_expanded)
    write_json(
        config.processed_dir / "doc2query_summary.json",
        {
            "generator": "offline_legal_keyphrase_fallback",
            "cases_expanded": len(cases_expanded),
            "statutes_expanded": len(statutes_expanded),
            "note": "Install a local seq2seq model later to replace generation without changing pipeline outputs.",
        },
    )
    return [
        build_bm25_run(
            cases_expanded,
            queries,
            top_k=config.case_top_k,
            run_id="doc2query_bm25_cases",
            index_path=config.indexes_dir / "doc2query_bm25_cases.pkl",
            run_path=config.runs_dir / "doc2query_bm25_cases.trec",
        ),
        build_bm25_run(
            statutes_expanded,
            queries,
            top_k=config.statute_top_k,
            run_id="doc2query_bm25_statutes",
            index_path=config.indexes_dir / "doc2query_bm25_statutes.pkl",
            run_path=config.runs_dir / "doc2query_bm25_statutes.trec",
        ),
    ]
