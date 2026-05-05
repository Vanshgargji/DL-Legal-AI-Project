"""ColBERT-compatible late-interaction fallback implemented without ML deps."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

from .config import PipelineConfig
from .io_utils import read_jsonl, write_json
from .text import simple_stem, tokenize
from .trec import read_run, write_run


class LateInteractionIndex:
    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs
        self.doc_ids = [doc["doc_id"] for doc in docs]
        self.doc_terms = []
        self.doc_stems = []
        self.doc_penalties = []
        self.exact_postings: dict[str, set[int]] = defaultdict(set)
        self.stem_postings: dict[str, set[int]] = defaultdict(set)
        self.prefix_postings: dict[str, set[int]] = defaultdict(set)
        df = Counter()
        for idx, doc in enumerate(docs):
            terms = set(tokenize(doc["text"], remove_stopwords=True))
            stems = {simple_stem(term) for term in terms}
            self.doc_terms.append(terms)
            self.doc_stems.append(stems)
            self.doc_penalties.append(1.0 / math.sqrt(1.0 + len(terms) / 3500.0))
            for term in terms:
                df[term] += 1
                self.exact_postings[term].add(idx)
                self.stem_postings[simple_stem(term)].add(idx)
                if len(term) >= 6:
                    self.prefix_postings[term[:5]].add(idx)
        n = max(1, len(docs))
        self.idf = {term: math.log(1.0 + n / (1 + freq)) for term, freq in df.items()}
        self.default_idf = math.log(1.0 + n)

    def _term_match(self, query_term: str, doc_idx: int) -> float:
        terms = self.doc_terms[doc_idx]
        if query_term in terms:
            return 1.0
        stem = simple_stem(query_term)
        if stem in self.doc_stems[doc_idx]:
            return 0.85
        if len(query_term) >= 6:
            prefix = query_term[:5]
            if any(term.startswith(prefix) for term in terms):
                return 0.55
        return 0.0

    def score(self, query: str, doc_idx: int) -> float:
        qterms = tokenize(query, remove_stopwords=True)
        if not qterms:
            return 0.0
        total = 0.0
        norm = 0.0
        for term in qterms:
            weight = self.idf.get(term, self.default_idf)
            total += weight * self._term_match(term, doc_idx)
            norm += weight
        return (total / norm if norm else 0.0) * self.doc_penalties[doc_idx]

    def search(self, query: str, top_k: int, candidates: list[str] | None = None) -> list[tuple[str, float]]:
        if candidates is None:
            qterms = tokenize(query, remove_stopwords=True)
            scores: dict[int, float] = defaultdict(float)
            norm = 0.0
            for term in qterms:
                weight = self.idf.get(term, self.default_idf)
                norm += weight
                term_scores: dict[int, float] = {}
                for idx in self.exact_postings.get(term, ()):
                    term_scores[idx] = 1.0
                stem = simple_stem(term)
                for idx in self.stem_postings.get(stem, ()):
                    if term_scores.get(idx, 0.0) < 0.85:
                        term_scores[idx] = 0.85
                if len(term) >= 6:
                    for idx in self.prefix_postings.get(term[:5], ()):
                        if term_scores.get(idx, 0.0) < 0.55:
                            term_scores[idx] = 0.55
                for idx, match in term_scores.items():
                    scores[idx] += weight * match
            scored = [
                (self.doc_ids[idx], (score / norm if norm else 0.0) * self.doc_penalties[idx])
                for idx, score in scores.items()
            ]
            scored.sort(key=lambda item: (-item[1], item[0]))
            return scored[:top_k]
        else:
            id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
            candidate_indices = [id_to_idx[doc_id] for doc_id in candidates if doc_id in id_to_idx]
            scored = [(self.doc_ids[idx], self.score(query, idx)) for idx in candidate_indices]
            scored.sort(key=lambda item: (-item[1], item[0]))
            return scored[:top_k]


def _existing_result(run_id: str, run_path: Path, expected_rows: int | None = None) -> dict | None:
    if not run_path.exists():
        return None
    rows = sum(1 for _ in run_path.open("r", encoding="utf-8"))
    if expected_rows is not None and rows != expected_rows:
        return None
    return {"run_id": run_id, "run_path": str(run_path), "rows": rows, "cached": True}


def _direct_run(index: LateInteractionIndex, queries: list[dict], top_k: int, run_id: str, run_path: Path) -> dict:
    existing = _existing_result(run_id, run_path, expected_rows=len(queries) * top_k)
    if existing:
        return existing
    rows = []
    for query in queries:
        ranked = index.search(query["text"], top_k)
        seen = {doc_id for doc_id, _ in ranked}
        if len(ranked) < top_k:
            for doc_id in index.doc_ids:
                if doc_id not in seen:
                    ranked.append((doc_id, 0.0))
                    seen.add(doc_id)
                    if len(ranked) == top_k:
                        break
        for rank, (doc_id, score) in enumerate(ranked[:top_k], start=1):
            rows.append({"query_id": query["query_id"], "doc_id": doc_id, "rank": rank, "score": score, "run_id": run_id})
    write_run(run_path, rows)
    return {"run_id": run_id, "run_path": str(run_path), "rows": len(rows)}


def _rerank_run(
    index: LateInteractionIndex,
    queries: list[dict],
    candidate_run_path: Path,
    top_k: int,
    depth: int,
    run_id: str,
    run_path: Path,
) -> dict:
    existing = _existing_result(run_id, run_path)
    if existing:
        existing["candidate_depth"] = depth
        return existing
    candidate_rows = read_run(candidate_run_path)
    by_query: dict[str, list[dict]] = defaultdict(list)
    for row in candidate_rows:
        by_query[row["query_id"]].append(row)
    for rows in by_query.values():
        rows.sort(key=lambda row: row["rank"])
    rows_out = []
    for query in queries:
        candidates = [row["doc_id"] for row in by_query.get(query["query_id"], [])[:depth]]
        ranked = index.search(query["text"], min(top_k, len(candidates)), candidates)
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            rows_out.append({"query_id": query["query_id"], "doc_id": doc_id, "rank": rank, "score": score, "run_id": run_id})
    write_run(run_path, rows_out)
    return {"run_id": run_id, "run_path": str(run_path), "rows": len(rows_out), "candidate_depth": depth}


def run_colbert_fallback(config: PipelineConfig) -> list[dict]:
    cases = read_jsonl(config.processed_dir / "cases.jsonl")
    statutes = read_jsonl(config.processed_dir / "statutes.jsonl")
    queries = read_jsonl(config.processed_dir / "queries.jsonl")
    case_index = LateInteractionIndex(cases)
    statute_index = LateInteractionIndex(statutes)
    outputs = [
        _direct_run(case_index, queries, config.case_top_k, "colbert_cases", config.runs_dir / "colbert_cases.trec"),
        _direct_run(statute_index, queries, config.statute_top_k, "colbert_statutes", config.runs_dir / "colbert_statutes.trec"),
        _rerank_run(
            case_index,
            queries,
            config.runs_dir / "bm25_cases.trec",
            config.case_top_k,
            config.rerank_depth,
            "bm25_colbert_cases",
            config.runs_dir / "bm25_colbert_cases.trec",
        ),
        _rerank_run(
            statute_index,
            queries,
            config.runs_dir / "bm25_statutes.trec",
            config.statute_top_k,
            config.rerank_depth,
            "bm25_colbert_statutes",
            config.runs_dir / "bm25_colbert_statutes.trec",
        ),
        _rerank_run(
            case_index,
            queries,
            config.runs_dir / "doc2query_bm25_cases.trec",
            config.case_top_k,
            config.rerank_depth,
            "doc2query_colbert_cases",
            config.runs_dir / "doc2query_colbert_cases.trec",
        ),
        _rerank_run(
            statute_index,
            queries,
            config.runs_dir / "doc2query_bm25_statutes.trec",
            config.statute_top_k,
            config.rerank_depth,
            "doc2query_colbert_statutes",
            config.runs_dir / "doc2query_colbert_statutes.trec",
        ),
    ]
    write_json(
        config.indexes_dir / "colbert_fallback_manifest.json",
        {
            "implementation": "pure_python_late_interaction_fallback",
            "reason": "Deep learning dependencies are not installed in this local environment.",
            "outputs": outputs,
        },
    )
    return outputs
