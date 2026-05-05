"""Configuration for Pipeline 3 summarization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SummarizationConfig:
    project_root: Path
    processed_dir: Path
    runs_dir: Path
    classification_predictions_dir: Path
    output_dir: Path
    processed_output_dir: Path
    summaries_dir: Path
    query_summaries_dir: Path
    evaluation_dir: Path
    analytics_dir: Path
    best_case_run: str = "doc2query_bm25_cases"
    best_statute_run: str = "bm25_statutes"
    max_case_summary_sentences: int = 8
    max_section_sentences: int = 2
    query_top_cases: int = 5
    query_top_statutes: int = 5

    @classmethod
    def from_root(cls, project_root: Path | str) -> "SummarizationConfig":
        root = Path(project_root).resolve()
        output = root / "summarization_outputs"
        return cls(
            project_root=root,
            processed_dir=root / "data" / "processed",
            runs_dir=root / "runs",
            classification_predictions_dir=root / "classification_outputs" / "predictions",
            output_dir=output,
            processed_output_dir=output / "processed",
            summaries_dir=output / "summaries",
            query_summaries_dir=output / "query_summaries",
            evaluation_dir=output / "evaluation",
            analytics_dir=output / "analytics",
        )
