"""PyTorch embedding backend for the face recognition pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import torch

from facial_recognition.core import Embedding, FaceDetection, ImageArray, RecognitionBackend
from facial_recognition.torch_training import image_to_tensor, load_torch_checkpoint


@dataclass(slots=True)
class TorchEmbeddingBackend:
    """Recognition backend using an external aligner and a trained PyTorch embedder."""

    alignment_backend: RecognitionBackend
    checkpoint_path: Path
    device: str = "cpu"
    class_names: tuple[str, ...] = field(init=False)
    image_size: tuple[int, int] = field(init=False)
    threshold: float = field(init=False)
    _embedder: torch.nn.Module = field(init=False, repr=False)
    _torch_device: torch.device = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._torch_device = torch.device(self.device)
        checkpoint = load_torch_checkpoint(self.checkpoint_path, device=self._torch_device)
        self.class_names = checkpoint.class_names
        self.image_size = checkpoint.image_size
        self.threshold = checkpoint.threshold
        self._embedder = checkpoint.model.to(self._torch_device)
        self._embedder.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        alignment_backend: RecognitionBackend,
        device: str = "cpu",
    ) -> Self:
        """Builds a backend from a saved checkpoint."""
        return cls(
            alignment_backend=alignment_backend,
            checkpoint_path=checkpoint_path,
            device=device,
        )

    def detect(self, image: ImageArray) -> list[FaceDetection]:
        """Delegates face detection to the alignment backend."""
        return self.alignment_backend.detect(image)

    def align(self, image: ImageArray, detection: FaceDetection) -> ImageArray:
        """Delegates landmark alignment to the alignment backend."""
        return self.alignment_backend.align(image, detection)

    def extract(self, aligned_face: ImageArray) -> Embedding:
        """Extracts an embedding from an aligned face crop."""
        tensor = image_to_tensor(aligned_face, image_size=self.image_size).unsqueeze(0).to(self._torch_device)
        with torch.inference_mode():
            embedding = self._embedder(tensor).squeeze(0).detach().cpu().numpy()
        return np.ascontiguousarray(embedding.astype(np.float32, copy=False))
