"""Visual analytics and reporting without external plotting dependencies."""

from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .io_utils import read_json, write_csv, write_json
from .trec import read_run


def _bar_svg(rows: list[dict], metric: str, title: str) -> str:
    width = 980
    bar_h = 24
    gap = 10
    left = 230
    height = 70 + len(rows) * (bar_h + gap)
    max_v = max([float(row.get(metric, 0.0)) for row in rows] + [0.001])
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{title}">',
        '<style>text{font-family:Arial,sans-serif;font-size:13px}.title{font-size:18px;font-weight:700}.bar{fill:#2563eb}.axis{stroke:#94a3b8;stroke-width:1}</style>',
        f'<text class="title" x="0" y="24">{title}</text>',
        f'<line class="axis" x1="{left}" y1="42" x2="{width-40}" y2="42"/>',
    ]
    for i, row in enumerate(rows):
        y = 58 + i * (bar_h + gap)
        value = float(row.get(metric, 0.0))
        bar_w = int((width - left - 80) * value / max_v)
        label = row["run"]
        parts.append(f'<text x="0" y="{y+17}">{label}</text>')
        parts.append(f'<rect class="bar" x="{left}" y="{y}" width="{bar_w}" height="{bar_h}" rx="3"/>')
        parts.append(f'<text x="{left + bar_w + 8}" y="{y+17}">{value:.4f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _overlap(run_a: Path, run_b: Path, k: int = 10) -> float:
    if not run_a.exists() or not run_b.exists():
        return 0.0
    a_rows = read_run(run_a)
    b_rows = read_run(run_b)
    by_q_a = {}
    by_q_b = {}
    for row in a_rows:
        by_q_a.setdefault(row["query_id"], []).append(row)
    for row in b_rows:
        by_q_b.setdefault(row["query_id"], []).append(row)
    scores = []
    for qid in by_q_a:
        a = {r["doc_id"] for r in sorted(by_q_a[qid], key=lambda r: r["rank"])[:k]}
        b = {r["doc_id"] for r in sorted(by_q_b.get(qid, []), key=lambda r: r["rank"])[:k]}
        if a or b:
            scores.append(len(a & b) / len(a | b))
    return sum(scores) / len(scores) if scores else 0.0


def build_analytics(config: PipelineConfig) -> dict:
    rows = []
    for path in sorted(config.metrics_dir.glob("*.json")):
        data = read_json(path)
        overall = data["overall"]
        rows.append(
            {
                "run": path.stem,
                "collection": "cases" if path.stem.endswith("_cases") else "statutes",
                "map": overall.get("map", 0.0),
                "mrr": overall.get("mrr", 0.0),
                "p@5": overall.get("p@5", 0.0),
                "p@10": overall.get("p@10", 0.0),
                "recall@10": overall.get("recall@10", 0.0),
                "recall@50": overall.get("recall@50", 0.0),
                "recall@100": overall.get("recall@100", 0.0),
                "ndcg@10": overall.get("ndcg@10", 0.0),
                "ndcg@100": overall.get("ndcg@100", 0.0),
            }
        )
    rows.sort(key=lambda r: (r["collection"], -float(r["ndcg@10"]), r["run"]))
    write_csv(
        config.analytics_dir / "metrics_comparison.csv",
        rows,
        ["run", "collection", "map", "mrr", "p@5", "p@10", "recall@10", "recall@50", "recall@100", "ndcg@10", "ndcg@100"],
    )
    best_cases = max((r for r in rows if r["collection"] == "cases"), key=lambda r: r["ndcg@10"], default=None)
    best_statutes = max((r for r in rows if r["collection"] == "statutes"), key=lambda r: r["ndcg@10"], default=None)
    overlaps = {
        "cases_bm25_vs_colbert_top10": _overlap(config.runs_dir / "bm25_cases.trec", config.runs_dir / "colbert_cases.trec"),
        "statutes_bm25_vs_colbert_top10": _overlap(config.runs_dir / "bm25_statutes.trec", config.runs_dir / "colbert_statutes.trec"),
    }
    write_json(config.analytics_dir / "overlap_summary.json", overlaps)
    cases_rows = [r for r in rows if r["collection"] == "cases"]
    statute_rows = [r for r in rows if r["collection"] == "statutes"]
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AILA Pipeline 1 Analytics</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #111827; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f8fafc; }}
    .note {{ color: #475569; }}
    .panel {{ margin: 20px 0 34px; }}
  </style>
</head>
<body>
  <h1>AILA 2019 Pipeline 1 Analytics</h1>
  <p class="note">Best case run: {best_cases['run'] if best_cases else 'n/a'}.
  Best statute run: {best_statutes['run'] if best_statutes else 'n/a'}.</p>
  <div class="panel">{_bar_svg(cases_rows, 'map', 'Cases: MAP by Method')}</div>
  <div class="panel">{_bar_svg(cases_rows, 'ndcg@10', 'Cases: nDCG@10 by Method')}</div>
  <div class="panel">{_bar_svg(cases_rows, 'recall@100', 'Cases: Recall@100 by Method')}</div>
  <div class="panel">{_bar_svg(statute_rows, 'map', 'Statutes: MAP by Method')}</div>
  <div class="panel">{_bar_svg(statute_rows, 'ndcg@10', 'Statutes: nDCG@10 by Method')}</div>
  <div class="panel">{_bar_svg(statute_rows, 'recall@100', 'Statutes: Recall@100 by Method')}</div>
  <h2>Run Comparison</h2>
  <table>
    <tr><th>Run</th><th>Collection</th><th>MAP</th><th>MRR</th><th>P@10</th><th>Recall@100</th><th>nDCG@10</th><th>nDCG@100</th></tr>
    {''.join(f"<tr><td>{r['run']}</td><td>{r['collection']}</td><td>{r['map']:.4f}</td><td>{r['mrr']:.4f}</td><td>{r['p@10']:.4f}</td><td>{r['recall@100']:.4f}</td><td>{r['ndcg@10']:.4f}</td><td>{r['ndcg@100']:.4f}</td></tr>" for r in rows)}
  </table>
  <h2>Retrieval Overlap</h2>
  <p>Cases BM25 vs late-interaction top-10 Jaccard: {overlaps['cases_bm25_vs_colbert_top10']:.4f}</p>
  <p>Statutes BM25 vs late-interaction top-10 Jaccard: {overlaps['statutes_bm25_vs_colbert_top10']:.4f}</p>
</body>
</html>
"""
    (config.analytics_dir / "pipeline1_report.html").write_text(html, encoding="utf-8")
    summary = {"best_cases": best_cases, "best_statutes": best_statutes, "runs": len(rows), "overlap": overlaps}
    write_json(config.analytics_dir / "analytics_summary.json", summary)
    return summary
