"""Lightning training lifecycle for ArcFace face-recognition models."""

import lightning
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torch import nn
from torch.nn import functional

from facial_recognition.config.config import TrainConfig
from facial_recognition.losses.arcface import ArcFaceLoss
from facial_recognition.models.classifier import NormalizedClassifier
from facial_recognition.utils.artifacts import RunPaths


class FaceRecognitionModule(lightning.LightningModule):
    """Train an embedding backbone with a normalized ArcFace classifier."""

    def __init__(
        self,
        backbone: nn.Module,
        classifier: NormalizedClassifier,
        arcface: ArcFaceLoss,
        config: TrainConfig,
    ) -> None:
        """Initialize the model components without allocating data or devices."""
        super().__init__()
        self._backbone = backbone
        self._classifier = classifier
        self._arcface = arcface
        self._config = config

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return normalized face embeddings."""
        return self._backbone(images)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int) -> torch.Tensor:
        """Compute and log a classification loss for one training batch."""
        del batch_index
        images, labels = batch
        loss = self._loss(images, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int) -> torch.Tensor:
        """Compute and log the validation loss used for best-checkpoint selection."""
        del batch_index
        images, labels = batch
        loss = self._loss(images, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Create the baseline AdamW optimizer."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )

    def _loss(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate ArcFace cross-entropy for one labeled batch."""
        embeddings = self(images)
        logits = self._arcface(self._classifier(embeddings), labels)
        return functional.cross_entropy(logits, labels)


def build_trainer(config: TrainConfig, run_paths: RunPaths) -> lightning.Trainer:
    """Build a local-MLflow trainer saving best and final checkpoints."""
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_paths.checkpoints,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    logger = MLFlowLogger(
        experiment_name=config.mlflow_experiment_name,
        tracking_uri=config.mlflow_tracking_dir.resolve().as_uri(),
        run_name=config.run_name,
    )
    return lightning.Trainer(
        accelerator=config.accelerator,
        callbacks=[checkpoint_callback],
        default_root_dir=run_paths.root,
        logger=logger,
        max_epochs=config.epochs,
    )
