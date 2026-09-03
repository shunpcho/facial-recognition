"""Tests for PyTorch training utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from facial_recognition import FaceDetection, RecognitionBackend
from facial_recognition.torch_backend import TorchEmbeddingBackend
from facial_recognition.torch_training import (
    FaceEmbeddingNet,
    PreparedFaceSample,
    TrainingConfig,
    load_torch_checkpoint,
    prepare_face_training_samples,
    save_torch_checkpoint,
    train_face_embedding_model,
)


class FakeAlignmentBackend(RecognitionBackend):
    """Deterministic backend for alignment-dependent PyTorch tests."""

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        del image
        return [
            FaceDetection(
                bounding_box=(0, 0, 4, 4),
                confidence=1.0,
                landmarks=((0.0, 0.0), (3.0, 0.0), (1.5, 1.5), (0.0, 3.0), (3.0, 3.0)),
            )
        ]

    def align(self, image: np.ndarray, detection: FaceDetection) -> np.ndarray:
        del detection
        return image

    def extract(self, aligned_face: np.ndarray) -> np.ndarray:
        del aligned_face
        msg = "This backend is only used for detection and alignment."
        raise NotImplementedError(msg)


def _write_image(image_path: Path, value: int) -> None:
    image = np.full((8, 8, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)


def test_prepare_face_training_samples_from_directory_tree(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    alice_dir = dataset_dir / "alice"
    bob_dir = dataset_dir / "bob"
    alice_dir.mkdir(parents=True)
    bob_dir.mkdir(parents=True)
    _write_image(alice_dir / "1.png", 32)
    _write_image(bob_dir / "1.png", 224)

    samples, class_names = prepare_face_training_samples(
        dataset_dir=dataset_dir,
        alignment_backend=FakeAlignmentBackend(),
        image_size=(6, 6),
    )

    assert class_names == ("alice", "bob")
    assert len(samples) == 2
    assert samples[0].identity == "alice"
    assert samples[0].label == 0
    assert tuple(samples[0].image.shape) == (3, 6, 6)
    assert samples[0].image.dtype == torch.float32


def test_train_face_embedding_model_returns_epoch_metrics() -> None:
    dark_face = torch.zeros((3, 8, 8), dtype=torch.float32)
    bright_face = torch.ones((3, 8, 8), dtype=torch.float32)
    samples = [
        PreparedFaceSample(identity="alice", label=0, image=dark_face),
        PreparedFaceSample(identity="alice", label=0, image=dark_face + 0.05),
        PreparedFaceSample(identity="bob", label=1, image=bright_face),
        PreparedFaceSample(identity="bob", label=1, image=bright_face - 0.05),
    ]

    model, metrics = train_face_embedding_model(
        samples=samples,
        num_classes=2,
        config=TrainingConfig(epochs=3, batch_size=2, embedding_dim=16, random_seed=7),
    )

    embedding = model(dark_face.unsqueeze(0))
    assert len(metrics) == 3
    assert metrics[-1].loss >= 0.0
    assert 0.0 <= metrics[-1].accuracy <= 1.0
    assert embedding.shape == (1, 16)
    assert torch.linalg.vector_norm(embedding, dim=1).item() == pytest.approx(1.0, abs=1e-5)


def test_torch_embedding_backend_loads_checkpoint_and_extracts_embeddings(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    model = FaceEmbeddingNet(embedding_dim=8)
    save_torch_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        class_names=("alice", "bob"),
        image_size=(8, 8),
        threshold=0.42,
    )

    checkpoint = load_torch_checkpoint(checkpoint_path)
    backend = TorchEmbeddingBackend.from_checkpoint(
        checkpoint_path=checkpoint_path,
        alignment_backend=FakeAlignmentBackend(),
    )

    embedding = backend.extract(np.full((8, 8, 3), 120, dtype=np.uint8))

    assert checkpoint.class_names == ("alice", "bob")
    assert checkpoint.threshold == pytest.approx(0.42)
    assert embedding.shape == (8,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)
