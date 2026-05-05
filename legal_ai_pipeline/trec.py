"""TREC run-file parsing, writing, and validation."""

from __future__ import annotations

from pathlib import Path

from .io_utils import ensure_dir


def write_run(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(
                f"{row['query_id']} Q0 {row['doc_id']} {int(row['rank'])} "
                f"{float(row['score']):.12f} {row['run_id']}\n"
            )


def read_run(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 6:
            raise ValueError(f"{path} line {line_no}: expected 6 columns, got {len(parts)}")
        rows.append(
            {
                "query_id": parts[0],
                "iter": parts[1],
                "doc_id": parts[2],
                "rank": int(parts[3]),
                "score": float(parts[4]),
                "run_id": parts[5],
            }
        )
    return rows


def validate_run(path: Path) -> dict:
    rows = read_run(path)
    problems = []
    last_qid = None
    seen_ranks: dict[str, set[int]] = {}
    for row in rows:
        if row["iter"] != "Q0":
            problems.append(f"{row['query_id']} rank {row['rank']} has iter={row['iter']}")
        seen_ranks.setdefault(row["query_id"], set())
        if row["rank"] in seen_ranks[row["query_id"]]:
            problems.append(f"duplicate rank {row['rank']} for {row['query_id']}")
        seen_ranks[row["query_id"]].add(row["rank"])
        last_qid = row["query_id"]
    return {
        "path": str(path),
        "rows": len(rows),
        "queries": len(seen_ranks),
        "valid": not problems,
        "problems": problems[:50],
        "last_query_seen": last_qid,
    }
