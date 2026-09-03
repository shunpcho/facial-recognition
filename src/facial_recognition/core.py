"""Core face recognition pipeline primitives."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

ImageArray = npt.NDArray[np.uint8]
Embedding = npt.NDArray[np.float32]
RawDetectionArray = npt.NDArray[np.float32]
Landmark = tuple[float, float]
GRAYSCALE_DIMENSIONS = 2
CHANNELS_AXIS = 2
SINGLE_CHANNEL = 1
BGR_CHANNELS = 3
BGRA_CHANNELS = 4


def _as_float32_vector(values: npt.ArrayLike) -> Embedding:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        msg = "Embedding vector must not be empty."
        raise ValueError(msg)
    return vector


def _l2_normalize(values: npt.ArrayLike) -> Embedding:
    vector = _as_float32_vector(values)
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float32).eps:
        msg = "Embedding vector norm must be greater than zero."
        raise ValueError(msg)
    return vector / norm


def preprocess_image(image: npt.ArrayLike) -> ImageArray:
    """Converts arbitrary image input into a contiguous BGR uint8 image.

    Raises:
        ValueError: If the input array is not grayscale, BGR, or BGRA.
    """
    array = np.asarray(image)
    if array.ndim == GRAYSCALE_DIMENSIONS:
        array = np.repeat(array[:, :, np.newaxis], BGR_CHANNELS, axis=CHANNELS_AXIS)
    elif array.ndim == BGR_CHANNELS and array.shape[CHANNELS_AXIS] == SINGLE_CHANNEL:
        array = np.repeat(array, BGR_CHANNELS, axis=CHANNELS_AXIS)
    elif array.ndim == BGR_CHANNELS and array.shape[CHANNELS_AXIS] == BGRA_CHANNELS:
        array = array[:, :, :BGR_CHANNELS]
    elif array.ndim != BGR_CHANNELS or array.shape[CHANNELS_AXIS] != BGR_CHANNELS:
        msg = "Expected a grayscale, BGR, or BGRA image array."
        raise ValueError(msg)

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.max(array, initial=0.0))
        scale = 255.0 if max_value <= 1.0 else 1.0
        array = np.clip(array * scale, 0.0, 255.0)
    else:
        array = np.clip(array, 0, 255)

    return np.ascontiguousarray(array.astype(np.uint8))


def cosine_similarity(left: npt.ArrayLike, right: npt.ArrayLike) -> float:
    """Computes cosine similarity between two embedding vectors.

    Raises:
        ValueError: If an embedding is empty, zero-norm, or shape-mismatched.
    """
    left_vector = _l2_normalize(left)
    right_vector = _l2_normalize(right)
    if left_vector.shape != right_vector.shape:
        msg = "Embedding vectors must have identical shapes."
        raise ValueError(msg)
    return float(np.dot(left_vector, right_vector))


@dataclass(frozen=True, slots=True)
class FaceDetection:
    """Single detected face region with landmarks."""

    bounding_box: tuple[int, int, int, int]
    confidence: float
    landmarks: tuple[Landmark, ...]
    raw_detection: RawDetectionArray | None = field(default=None, repr=False)

    @property
    def area(self) -> int:
        """Face bounding-box area."""
        return self.bounding_box[2] * self.bounding_box[3]


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Pairwise face comparison result."""

    similarity: float
    threshold: float
    accepted: bool


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Face identification result against a gallery."""

    identity: str | None
    similarity: float
    threshold: float
    accepted: bool


class RecognitionBackend(Protocol):
    """Abstract backend for detection, alignment, and feature extraction."""

    def detect(self, image: ImageArray) -> list[FaceDetection]:
        """Detects faces from a BGR uint8 image."""
        ...

    def align(self, image: ImageArray, detection: FaceDetection) -> ImageArray:
        """Aligns a detected face to a canonical crop."""
        ...

    def extract(self, aligned_face: ImageArray) -> Embedding:
        """Extracts an embedding vector from an aligned face crop."""
        ...


@dataclass(slots=True)
class FaceGallery:
    """In-memory identity gallery backed by normalized embeddings."""

    _embeddings: dict[str, list[Embedding]] = field(default_factory=lambda: defaultdict(list))

    def add_embedding(self, identity: str, embedding: npt.ArrayLike) -> None:
        """Stores a normalized embedding for an identity."""
        normalized = _l2_normalize(embedding)
        self._embeddings[identity].append(normalized)

    def add_embeddings(self, identity: str, embeddings: Sequence[npt.ArrayLike]) -> None:
        """Stores multiple embeddings for an identity."""
        for embedding in embeddings:
            self.add_embedding(identity, embedding)

    def identities(self) -> tuple[str, ...]:
        """Returns registered identities."""
        return tuple(sorted(self._embeddings))

    def has_identity(self, identity: str) -> bool:
        """Returns whether the identity exists in the gallery."""
        return identity in self._embeddings

    def prototype(self, identity: str) -> Embedding:
        """Returns an L2-normalized prototype vector for an identity.

        Raises:
            KeyError: If the identity is not registered.
        """
        if identity not in self._embeddings:
            msg = f"Identity not found: {identity}"
            raise KeyError(msg)
        stacked = np.vstack(self._embeddings[identity]).astype(np.float32, copy=False)
        return _l2_normalize(np.mean(stacked, axis=0))

    def compare(self, identity: str, embedding: npt.ArrayLike, threshold: float) -> MatchResult:
        """Compares an embedding with a claimed identity."""
        similarity = cosine_similarity(self.prototype(identity), embedding)
        return MatchResult(
            identity=identity,
            similarity=similarity,
            threshold=threshold,
            accepted=similarity >= threshold,
        )

    def best_match(self, embedding: npt.ArrayLike, threshold: float) -> MatchResult:
        """Finds the best identity match for an embedding."""
        normalized = _l2_normalize(embedding)
        best_identity: str | None = None
        best_similarity = -1.0
        for identity in self.identities():
            similarity = cosine_similarity(self.prototype(identity), normalized)
            if similarity > best_similarity:
                best_identity = identity
                best_similarity = similarity
        accepted = best_identity is not None and best_similarity >= threshold
        return MatchResult(
            identity=best_identity if accepted else None,
            similarity=best_similarity,
            threshold=threshold,
            accepted=accepted,
        )


@dataclass(slots=True)
class FaceRecognitionPipeline:
    """End-to-end pipeline for robust face recognition."""

    backend: RecognitionBackend
    threshold: float = 0.363
    gallery: FaceGallery = field(default_factory=FaceGallery)

    def detect_faces(self, image: npt.ArrayLike) -> list[FaceDetection]:
        """Runs face detection on an input image."""
        prepared_image = preprocess_image(image)
        return self.backend.detect(prepared_image)

    def _select_detection(self, detections: Sequence[FaceDetection]) -> FaceDetection:
        if not detections:
            msg = "No faces were detected in the image."
            raise ValueError(msg)
        return max(detections, key=lambda detection: detection.area)

    def encode(
        self,
        image: npt.ArrayLike,
        detection: FaceDetection | None = None,
    ) -> Embedding:
        """Detects, aligns, preprocesses, and encodes the most prominent face."""
        prepared_image = preprocess_image(image)
        selected_detection = detection or self._select_detection(self.backend.detect(prepared_image))
        aligned_face = self.backend.align(prepared_image, selected_detection)
        embedding = self.backend.extract(preprocess_image(aligned_face))
        return _l2_normalize(embedding)

    def enroll(
        self,
        identity: str,
        image: npt.ArrayLike,
        detection: FaceDetection | None = None,
    ) -> Embedding:
        """Encodes an image and stores it in the gallery."""
        embedding = self.encode(image, detection=detection)
        self.gallery.add_embedding(identity, embedding)
        return embedding

    def compare_images(
        self,
        reference_image: npt.ArrayLike,
        candidate_image: npt.ArrayLike,
    ) -> ComparisonResult:
        """Compares two face images directly."""
        reference_embedding = self.encode(reference_image)
        candidate_embedding = self.encode(candidate_image)
        similarity = cosine_similarity(reference_embedding, candidate_embedding)
        return ComparisonResult(
            similarity=similarity,
            threshold=self.threshold,
            accepted=similarity >= self.threshold,
        )

    def verify_identity(self, identity: str, image: npt.ArrayLike) -> MatchResult:
        """Verifies whether an image belongs to a claimed identity."""
        embedding = self.encode(image)
        return self.gallery.compare(identity, embedding, threshold=self.threshold)

    def recognize(self, image: npt.ArrayLike) -> MatchResult:
        """Finds the best gallery match for an image."""
        embedding = self.encode(image)
        return self.gallery.best_match(embedding, threshold=self.threshold)
