"""Tests for the Lightning model's training contract."""

from pathlib import Path

import torch
from torch import nn

from facial_recognition.config.config import TrainConfig
from facial_recognition.engine.trainer import FaceRecognitionModule
from facial_recognition.losses.arcface import ArcFaceLoss
from facial_recognition.models.classifier import NormalizedClassifier


class _Backbone(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.ones((images.shape[0], 512), dtype=images.dtype)


def test_training_step_returns_scalar_loss() -> None:
    config = TrainConfig(metadata_path=Path("metadata.csv"))
    model = FaceRecognitionModule(_Backbone(), NormalizedClassifier(2), ArcFaceLoss(), config)

    loss = model.training_step((torch.randn(2, 3, 112, 112), torch.tensor([0, 1])), batch_index=0)

    assert loss.ndim == 0
