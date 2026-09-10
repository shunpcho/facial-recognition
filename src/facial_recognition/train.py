"""Command-line entry point for IR50 and ArcFace training."""

from pathlib import Path

import mlflow
from cyclopts import App
from torch.utils.data import DataLoader

from facial_recognition.config.config import Accelerator, TrainConfig
from facial_recognition.datasets.face_dataset import FaceDataset
from facial_recognition.datasets.metadata import build_identity_mapping, load_metadata, records_for_split
from facial_recognition.datasets.transforms import build_transform
from facial_recognition.engine.trainer import build_trainer, FaceRecognitionModule
from facial_recognition.losses.arcface import ArcFaceLoss
from facial_recognition.models.classifier import NormalizedClassifier
from facial_recognition.models.ir50 import IR50
from facial_recognition.utils.artifacts import RunPaths
from facial_recognition.utils.mlflow import configure_local_tracking
from facial_recognition.utils.seed import seed_everything

app = App()


@app.default
def run(
    metadata_path: Path,
    run_name: str = "baseline",
    output_dir: Path = Path("outputs"),
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    accelerator: Accelerator = "auto",
) -> None:
    """Train a randomly initialized IR50 ArcFace baseline on pre-aligned faces."""
    config = TrainConfig(
        metadata_path=metadata_path,
        run_name=run_name,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        accelerator=accelerator,
    )
    seed_everything(config.seed)
    records = load_metadata(config.metadata_path)
    train_records = records_for_split(records, "train")
    validation_records = records_for_split(records, "val")
    identity_mapping = build_identity_mapping(train_records)
    train_dataset = FaceDataset(train_records, identity_mapping, build_transform("train"))
    validation_dataset = FaceDataset(validation_records, identity_mapping, build_transform("validation"))

    run_paths = RunPaths.create(config.output_dir, config.run_name)
    config_path = run_paths.write_config(config.as_serializable_dict())
    mapping_path = run_paths.write_artifact_json("identity_mapping.json", identity_mapping)
    configure_local_tracking(config.mlflow_tracking_dir, config.mlflow_experiment_name)
    model = FaceRecognitionModule(
        IR50(),
        NormalizedClassifier(num_classes=len(identity_mapping)),
        ArcFaceLoss(),
        config,
    )
    trainer = build_trainer(config, run_paths)
    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params(config.as_serializable_dict())
        mlflow.log_artifact(str(config_path))
        mlflow.log_artifact(str(mapping_path))
        trainer.fit(
            model,
            DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
            DataLoader(validation_dataset, batch_size=config.batch_size),
        )


def main() -> None:
    """Run the Cyclopts command-line application."""
    app()
