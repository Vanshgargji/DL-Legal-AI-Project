"""Configuration for the final model benchmarking pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelBenchmarkConfig:
    project_root: Path
    processed_dir: Path
    baseline_runs_dir: Path
    baseline_metrics_dir: Path
    classification_predictions_dir: Path
    summarization_dir: Path
    rag_dir: Path
    output_dir: Path
    models_dir: Path
    embeddings_dir: Path
    runs_dir: Path
    classification_dir: Path
    summarization_output_dir: Path
    rag_output_dir: Path
    metrics_dir: Path
    analytics_dir: Path
    top_k_cases: int = 1000
    top_k_statutes: int = 197
    rerank_depth: int = 200
    smoke_limit_docs: int = 250

    @classmethod
    def from_root(cls, project_root: Path | str) -> "ModelBenchmarkConfig":
        root = Path(project_root).resolve()
        output = root / "model_benchmark_outputs"
        return cls(
            project_root=root,
            processed_dir=root / "data" / "processed",
            baseline_runs_dir=root / "runs",
            baseline_metrics_dir=root / "metrics",
            classification_predictions_dir=root / "classification_outputs" / "predictions",
            summarization_dir=root / "summarization_outputs",
            rag_dir=root / "rag_outputs",
            output_dir=output,
            models_dir=output / "models",
            embeddings_dir=output / "embeddings",
            runs_dir=output / "runs",
            classification_dir=output / "classification",
            summarization_output_dir=output / "summarization",
            rag_output_dir=output / "rag",
            metrics_dir=output / "metrics",
            analytics_dir=output / "analytics",
        )
