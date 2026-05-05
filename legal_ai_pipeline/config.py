"""Configuration defaults for Pipeline 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    dataset_dir: Path
    processed_dir: Path
    indexes_dir: Path
    runs_dir: Path
    metrics_dir: Path
    rag_dir: Path
    analytics_dir: Path
    logs_dir: Path
    case_top_k: int = 1000
    statute_top_k: int = 197
    rerank_depth: int = 200
    rag_top_cases: int = 5
    rag_top_statutes: int = 5

    @classmethod
    def from_root(cls, project_root: Path | str) -> "PipelineConfig":
        root = Path(project_root).resolve()
        dataset = root / "dataset aila"
        return cls(
            project_root=root,
            dataset_dir=dataset,
            processed_dir=root / "data" / "processed",
            indexes_dir=root / "indexes",
            runs_dir=root / "runs",
            metrics_dir=root / "metrics",
            rag_dir=root / "rag_outputs",
            analytics_dir=root / "analytics",
            logs_dir=root / "logs",
        )

    @property
    def case_docs_dir(self) -> Path:
        return self.dataset_dir / "Object_casedocs"

    @property
    def statute_docs_dir(self) -> Path:
        return self.dataset_dir / "Object_statutes"

    @property
    def query_file(self) -> Path:
        return self.dataset_dir / "Query_doc.txt"

    @property
    def case_qrels_file(self) -> Path:
        return self.dataset_dir / "relevance_judgments_priorcases.txt"

    @property
    def statute_qrels_file(self) -> Path:
        return self.dataset_dir / "relevance_judgments_statutes.txt"
