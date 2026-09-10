"""Experiment configuration shared by training and evaluation entry points."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Accelerator = Literal["auto", "cpu", "gpu"]


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Configuration for a reproducible face-recognition training run.

    Attributes:
        metadata_path: CSV metadata containing image paths, identities, and splits.
        output_dir: Directory containing all named run directories.
        run_name: Stable identifier for this run under ``output_dir``.
        epochs: Number of training epochs.
        batch_size: Number of samples processed in one optimizer step.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        seed: Random seed persisted with the run configuration.
        accelerator: Lightning accelerator selection.
        mlflow_experiment_name: Local MLflow experiment name.
        mlflow_tracking_dir: Directory for file-backed MLflow tracking data.
    """

    metadata_path: Path
    output_dir: Path = Path("outputs")
    run_name: str = "baseline"
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    accelerator: Accelerator = "auto"
    mlflow_experiment_name: str = "face_recognition"
    mlflow_tracking_dir: Path = Path("mlruns")

    def __post_init__(self) -> None:
        """Validate values that would otherwise fail later during training.

        Raises:
            ValueError: If a training hyperparameter is invalid.
        """
        if not self.run_name:
            msg = "run_name must not be empty"
            raise ValueError(msg)
        if self.epochs <= 0:
            msg = "epochs must be positive"
            raise ValueError(msg)
        if self.batch_size <= 0:
            msg = "batch_size must be positive"
            raise ValueError(msg)
        if self.learning_rate <= 0:
            msg = "learning_rate must be positive"
            raise ValueError(msg)
        if self.weight_decay < 0:
            msg = "weight_decay must be non-negative"
            raise ValueError(msg)

    def as_serializable_dict(self) -> dict[str, str | int | float]:
        """Return a JSON-compatible representation of this resolved config."""
        values = asdict(self)
        return {name: str(value) if isinstance(value, Path) else value for name, value in values.items()}
