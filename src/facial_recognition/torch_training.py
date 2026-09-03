"""PyTorch training utilities for face embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from facial_recognition.core import FaceDetection, ImageArray, RecognitionBackend, preprocess_image

RGB_CHANNELS = 3
DEFAULT_IMAGE_SIZE = (112, 112)
MAX_UINT8_VALUE = 255.0
SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _select_largest_face(detections: Sequence[FaceDetection]) -> FaceDetection:
    if not detections:
        msg = "No faces were detected in the image."
        raise ValueError(msg)
    return max(detections, key=lambda detection: detection.area)


def load_image_file(image_path: Path) -> ImageArray:
    """Loads an image from disk as a BGR uint8 array.

    Raises:
        ValueError: If the image cannot be decoded.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        msg = f"Could not decode image: {image_path}"
        raise ValueError(msg)
    return preprocess_image(image)


def image_to_tensor(image: npt.ArrayLike, image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> Tensor:
    """Converts an image array into a normalized CHW float32 tensor."""
    prepared = preprocess_image(image)
    resized = cv2.resize(prepared, image_size, interpolation=cv2.INTER_AREA)
    channels_first = np.transpose(resized, (2, 0, 1)).astype(np.float32, copy=False)
    return torch.from_numpy(np.ascontiguousarray(channels_first / MAX_UINT8_VALUE))


@dataclass(frozen=True, slots=True)
class PreparedFaceSample:
    """Aligned tensor sample paired with an identity label."""

    identity: str
    label: int
    image: Tensor


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters for PyTorch face-embedding training."""

    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    embedding_dim: int = 128
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    random_seed: int = 0
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Aggregated metrics for one training epoch."""

    epoch: int
    loss: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class TrainedCheckpoint:
    """Loaded PyTorch checkpoint state."""

    model: FaceEmbeddingNet
    class_names: tuple[str, ...]
    image_size: tuple[int, int]
    threshold: float


class PreparedFaceDataset(Dataset[tuple[Tensor, int]]):
    """Dataset built from pre-aligned in-memory face tensors."""

    def __init__(self, samples: Sequence[PreparedFaceSample]) -> None:
        self._samples = list(samples)

    def __len__(self) -> int:
        """Dataset length."""
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Returns one tensor-label pair."""
        sample = self._samples[index]
        return sample.image, sample.label


class FaceEmbeddingNet(nn.Module):
    """Compact convolutional network for face embeddings."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.features = nn.Sequential(
            nn.Conv2d(RGB_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(128, embedding_dim)

    def forward(self, images: Tensor) -> Tensor:
        """Returns L2-normalized embeddings for a batch of images."""
        features = self.features(images)
        flattened = torch.flatten(features, start_dim=1)
        projected = self.projection(flattened)
        return functional.normalize(projected, p=2.0, dim=1)


class FaceClassificationModel(nn.Module):
    """Classifier head used to train the embedding model."""

    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.embedder = FaceEmbeddingNet(embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        """Returns identity logits for a batch of images."""
        embeddings = self.embedder(images)
        return self.classifier(embeddings)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _collect_identity_directories(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.is_dir():
        msg = f"Dataset directory does not exist: {dataset_dir}"
        raise FileNotFoundError(msg)
    directories = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if not directories:
        msg = f"No identity directories found under: {dataset_dir}"
        raise ValueError(msg)
    return directories


def _collect_identity_image_paths(identity_dir: Path) -> list[Path]:
    image_paths = sorted(
        path
        for path in identity_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not image_paths:
        msg = f"No image files found under: {identity_dir}"
        raise ValueError(msg)
    return image_paths


def prepare_face_training_samples(
    dataset_dir: Path,
    alignment_backend: RecognitionBackend,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> tuple[list[PreparedFaceSample], tuple[str, ...]]:
    """Loads, aligns, and tensorizes a directory-organized face dataset."""
    samples: list[PreparedFaceSample] = []
    class_names: list[str] = []

    for label, identity_dir in enumerate(_collect_identity_directories(dataset_dir)):
        class_names.append(identity_dir.name)
        for image_path in _collect_identity_image_paths(identity_dir):
            image = load_image_file(image_path)
            detection = _select_largest_face(alignment_backend.detect(image))
            aligned_face = alignment_backend.align(image, detection)
            samples.append(
                PreparedFaceSample(
                    identity=identity_dir.name,
                    label=label,
                    image=image_to_tensor(aligned_face, image_size=image_size),
                )
            )

    if len(class_names) < 2:
        msg = "PyTorch training requires at least two identity directories."
        raise ValueError(msg)

    return samples, tuple(class_names)


def train_face_embedding_model(
    samples: Sequence[PreparedFaceSample],
    num_classes: int,
    config: TrainingConfig,
) -> tuple[FaceEmbeddingNet, tuple[EpochMetrics, ...]]:
    """Trains a face embedding model with a classifier head."""
    if not samples:
        msg = "Training samples must not be empty."
        raise ValueError(msg)
    if num_classes < 2:
        msg = "Training requires at least two classes."
        raise ValueError(msg)

    torch.manual_seed(config.random_seed)
    device = _resolve_device(config.device)
    model = FaceClassificationModel(embedding_dim=config.embedding_dim, num_classes=num_classes).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.CrossEntropyLoss()
    dataset = PreparedFaceDataset(samples)
    generator = torch.Generator().manual_seed(config.random_seed)
    dataloader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )

    metrics: list[EpochMetrics] = []
    for epoch_index in range(config.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_examples = 0

        for images, labels in dataloader:
            batch_images = images.to(device)
            batch_labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_images)
            loss = loss_function(logits, batch_labels)
            loss.backward()
            optimizer.step()

            batch_size = int(batch_labels.shape[0])
            running_examples += batch_size
            running_loss += float(loss.detach().cpu().item()) * batch_size
            predictions = torch.argmax(logits, dim=1)
            running_correct += int((predictions == batch_labels).sum().detach().cpu().item())

        metrics.append(
            EpochMetrics(
                epoch=epoch_index + 1,
                loss=running_loss / running_examples,
                accuracy=running_correct / running_examples,
            )
        )

    model.eval()
    embedder = model.embedder.to(torch.device("cpu"))
    return embedder, tuple(metrics)


def save_torch_checkpoint(
    checkpoint_path: Path,
    model: FaceEmbeddingNet,
    class_names: Sequence[str],
    image_size: tuple[int, int],
    threshold: float,
) -> None:
    """Saves a trained face embedding model checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "embedding_dim": model.embedding_dim,
            "class_names": list(class_names),
            "image_size": list(image_size),
            "threshold": threshold,
        },
        checkpoint_path,
    )


def load_torch_checkpoint(
    checkpoint_path: Path,
    device: str | torch.device = "cpu",
) -> TrainedCheckpoint:
    """Loads a trained face embedding checkpoint from disk."""
    checkpoint = cast(
        "dict[str, Any]",
        torch.load(checkpoint_path, map_location=device, weights_only=False),
    )
    embedding_dim = int(checkpoint["embedding_dim"])
    image_size_values = cast("Sequence[int]", checkpoint["image_size"])
    if len(image_size_values) != 2:
        msg = "Checkpoint image_size must contain exactly two integers."
        raise ValueError(msg)
    model = FaceEmbeddingNet(embedding_dim=embedding_dim)
    model.load_state_dict(cast("dict[str, Tensor]", checkpoint["state_dict"]))
    model.eval()
    return TrainedCheckpoint(
        model=model,
        class_names=tuple(str(name) for name in cast("Sequence[object]", checkpoint["class_names"])),
        image_size=(int(image_size_values[0]), int(image_size_values[1])),
        threshold=float(checkpoint.get("threshold", 0.363)),
    )
