"""Dataset ingestion and validation for AILA 2019."""

from __future__ import annotations

import re
from pathlib import Path

from .config import PipelineConfig
from .io_utils import ensure_dir, write_csv, write_json, write_jsonl
from .text import normalize_text, token_count


ID_NUM_RE = re.compile(r"(\d+)")


def _numeric_sort_key(path: Path) -> tuple[int, str]:
    match = ID_NUM_RE.search(path.stem)
    return (int(match.group(1)) if match else 0, path.name)


def read_case_documents(config: PipelineConfig) -> list[dict]:
    docs = []
    for path in sorted(config.case_docs_dir.glob("C*.txt"), key=_numeric_sort_key):
        raw = path.read_text(encoding="utf-8", errors="replace")
        docs.append(
            {
                "doc_id": path.stem,
                "collection": "cases",
                "path": str(path),
                "title": "",
                "text": normalize_text(raw),
            }
        )
    return docs


def read_statute_documents(config: PipelineConfig) -> list[dict]:
    docs = []
    for path in sorted(config.statute_docs_dir.glob("S*.txt"), key=_numeric_sort_key):
        raw = path.read_text(encoding="utf-8", errors="replace")
        title = ""
        desc = raw
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if lines and lines[0].lower().startswith("title:"):
            title = lines[0].split(":", 1)[1].strip()
        if len(lines) > 1 and lines[1].lower().startswith("desc:"):
            desc = lines[1].split(":", 1)[1].strip()
        text = normalize_text((title + ". " + desc).strip())
        docs.append(
            {
                "doc_id": path.stem,
                "collection": "statutes",
                "path": str(path),
                "title": normalize_text(title),
                "text": text,
            }
        )
    return docs


def read_queries(config: PipelineConfig) -> list[dict]:
    queries = []
    for line in config.query_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        if "||" not in line:
            raise ValueError(f"Malformed query line: {line[:120]}")
        qid, text = line.split("||", 1)
        queries.append({"query_id": qid.strip(), "text": normalize_text(text)})
    return queries


def read_qrels(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed qrels line in {path.name}: {line}")
        rows.append(
            {
                "query_id": parts[0],
                "iter": parts[1],
                "doc_id": parts[2],
                "relevance": int(parts[3]),
            }
        )
    return rows


def ingest(config: PipelineConfig) -> dict:
    ensure_dir(config.processed_dir)
    for directory in (
        config.indexes_dir,
        config.runs_dir,
        config.metrics_dir,
        config.rag_dir,
        config.analytics_dir,
        config.logs_dir,
    ):
        ensure_dir(directory)

    cases = read_case_documents(config)
    statutes = read_statute_documents(config)
    queries = read_queries(config)
    case_qrels = read_qrels(config.case_qrels_file)
    statute_qrels = read_qrels(config.statute_qrels_file)

    metadata = []
    for doc in cases + statutes:
        metadata.append(
            {
                "doc_id": doc["doc_id"],
                "collection": doc["collection"],
                "path": doc["path"],
                "title": doc.get("title", ""),
                "token_count": token_count(doc["text"]),
            }
        )

    write_jsonl(config.processed_dir / "cases.jsonl", cases)
    write_jsonl(config.processed_dir / "statutes.jsonl", statutes)
    write_jsonl(config.processed_dir / "queries.jsonl", queries)
    write_jsonl(config.processed_dir / "qrels_cases.jsonl", case_qrels)
    write_jsonl(config.processed_dir / "qrels_statutes.jsonl", statute_qrels)
    write_csv(
        config.processed_dir / "document_metadata.csv",
        metadata,
        ["doc_id", "collection", "path", "title", "token_count"],
    )

    summary = {
        "cases": len(cases),
        "statutes": len(statutes),
        "queries": len(queries),
        "case_qrels_rows": len(case_qrels),
        "statute_qrels_rows": len(statute_qrels),
        "case_positive_qrels": sum(1 for r in case_qrels if r["relevance"] > 0),
        "statute_positive_qrels": sum(1 for r in statute_qrels if r["relevance"] > 0),
    }
    expected = {"cases": 2914, "statutes": 197, "queries": 50}
    summary["acceptance"] = {
        key: {"expected": value, "actual": summary[key], "passed": summary[key] == value}
        for key, value in expected.items()
    }
    write_json(config.processed_dir / "ingestion_summary.json", summary)
    return summary
