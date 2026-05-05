"""Pure-Python BM25 implementation and run generation."""

from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from .config import PipelineConfig
from .io_utils import ensure_dir, read_jsonl
from .text import tokenize
from .trec import write_run


class BM25Index:
    def __init__(self, docs: list[dict], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids = [doc["doc_id"] for doc in docs]
        self.doc_len: list[int] = []
        self.term_freqs: list[Counter] = []
        self.inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for idx, doc in enumerate(docs):
            terms = tokenize(doc["text"])
            counts = Counter(terms)
            self.term_freqs.append(counts)
            self.doc_len.append(len(terms))
            for term, freq in counts.items():
                self.inverted[term].append((idx, freq))

        self.num_docs = len(self.doc_ids)
        self.avgdl = sum(self.doc_len) / max(1, self.num_docs)
        self.idf = {
            term: math.log(1.0 + (self.num_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self.inverted.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        query_terms = tokenize(query)
        for term in query_terms:
            postings = self.inverted.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_idx, freq in postings:
                denom = freq + self.k1 * (1.0 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl)
                scores[doc_idx] += idf * (freq * (self.k1 + 1.0) / denom)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.doc_ids[item[0]]))
        return [(self.doc_ids[idx], score) for idx, score in ranked[:top_k]]


def save_index(path: Path, index: BM25Index) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(index, f)


def load_index(path: Path) -> BM25Index:
    with path.open("rb") as f:
        return pickle.load(f)


def build_bm25_run(
    docs: list[dict],
    queries: list[dict],
    *,
    top_k: int,
    run_id: str,
    index_path: Path,
    run_path: Path,
) -> dict:
    index = BM25Index(docs)
    save_index(index_path, index)
    rows = []
    for query in queries:
        for rank, (doc_id, score) in enumerate(index.search(query["text"], top_k), start=1):
            rows.append(
                {
                    "query_id": query["query_id"],
                    "doc_id": doc_id,
                    "rank": rank,
                    "score": score,
                    "run_id": run_id,
                }
            )
    write_run(run_path, rows)
    return {"run_id": run_id, "run_path": str(run_path), "index_path": str(index_path), "rows": len(rows)}


def run_bm25(config: PipelineConfig) -> list[dict]:
    cases = read_jsonl(config.processed_dir / "cases.jsonl")
    statutes = read_jsonl(config.processed_dir / "statutes.jsonl")
    queries = read_jsonl(config.processed_dir / "queries.jsonl")
    return [
        build_bm25_run(
            cases,
            queries,
            top_k=config.case_top_k,
            run_id="bm25_cases",
            index_path=config.indexes_dir / "bm25_cases.pkl",
            run_path=config.runs_dir / "bm25_cases.trec",
        ),
        build_bm25_run(
            statutes,
            queries,
            top_k=config.statute_top_k,
            run_id="bm25_statutes",
            index_path=config.indexes_dir / "bm25_statutes.pkl",
            run_path=config.runs_dir / "bm25_statutes.trec",
        ),
    ]
