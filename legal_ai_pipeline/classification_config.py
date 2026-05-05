"""Configuration for Pipeline 2 legal issue classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClassificationConfig:
    project_root: Path
    processed_dir: Path
    output_dir: Path
    labels_dir: Path
    features_dir: Path
    models_dir: Path
    predictions_dir: Path
    metrics_dir: Path
    analytics_dir: Path
    confidence_threshold: float = 0.18
    max_labels_per_item: int = 4

    @classmethod
    def from_root(cls, project_root: Path | str) -> "ClassificationConfig":
        root = Path(project_root).resolve()
        output = root / "classification_outputs"
        return cls(
            project_root=root,
            processed_dir=root / "data" / "processed",
            output_dir=output,
            labels_dir=output / "labels",
            features_dir=output / "features",
            models_dir=output / "models",
            predictions_dir=output / "predictions",
            metrics_dir=output / "metrics",
            analytics_dir=output / "analytics",
        )
