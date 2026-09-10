"""Local MLflow configuration helpers."""

from pathlib import Path

import mlflow


def configure_local_tracking(tracking_dir: Path, experiment_name: str) -> None:
    """Configure MLflow to record runs locally under ``tracking_dir``."""
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment(experiment_name)
