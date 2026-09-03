"""Tests for the face recognition pipeline core."""

from __future__ import annotations

import numpy as np
import pytest

from facial_recognition import (
    cosine_similarity,
    FaceDetection,
    FaceRecognitionPipeline,
    preprocess_image,
    RecognitionBackend,
)


class FakeBackend(RecognitionBackend):
    """Deterministic backend for pipeline tests."""

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        del image
        return [
            FaceDetection(
                bounding_box=(0, 0, 2, 2),
                confidence=0.99,
                landmarks=((0.0, 0.0), (1.0, 0.0), (0.5, 0.5), (0.0, 1.0), (1.0, 1.0)),
            )
        ]

    def align(self, image: np.ndarray, detection: FaceDetection) -> np.ndarray:
        del detection
        return image[:2, :2]

    def extract(self, aligned_face: np.ndarray) -> np.ndarray:
        mean_value = float(np.mean(aligned_face))
        return np.array([mean_value, mean_value / 2.0, 1.0], dtype=np.float32)


def test_preprocess_image_converts_grayscale_float_to_bgr_uint8() -> None:
    image = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)

    processed = preprocess_image(image)

    assert processed.dtype == np.uint8
    assert processed.shape == (2, 2, 3)
    np.testing.assert_array_equal(processed[0, 1], np.array([127, 127, 127], dtype=np.uint8))


def test_cosine_similarity_requires_matching_dimensions() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        cosine_similarity(np.array([1.0, 0.0], dtype=np.float32), np.array([1.0], dtype=np.float32))


def test_pipeline_enroll_and_recognize_identity() -> None:
    backend = FakeBackend()
    pipeline = FaceRecognitionPipeline(backend=backend, threshold=0.95)
    alice_image = np.full((4, 4, 3), 64, dtype=np.uint8)
    bob_image = np.full((4, 4, 3), 180, dtype=np.uint8)

    pipeline.enroll("alice", alice_image)
    pipeline.enroll("bob", bob_image)

    result = pipeline.recognize(np.full((4, 4, 3), 66, dtype=np.uint8))

    assert result.identity == "alice"
    assert result.threshold == 0.95
    assert result.accepted
    assert result.similarity == pytest.approx(0.9999999, abs=1e-6)


def test_pipeline_rejects_unknown_identity_when_below_threshold() -> None:
    backend = FakeBackend()
    pipeline = FaceRecognitionPipeline(backend=backend, threshold=0.99999)
    pipeline.enroll("alice", np.full((4, 4, 3), 64, dtype=np.uint8))

    result = pipeline.recognize(np.full((4, 4, 3), 180, dtype=np.uint8))

    assert result.identity is None
    assert not result.accepted
    assert result.similarity < result.threshold


def test_compare_images_uses_cosine_similarity_threshold() -> None:
    backend = FakeBackend()
    pipeline = FaceRecognitionPipeline(backend=backend, threshold=0.9)

    result = pipeline.compare_images(
        np.full((4, 4, 3), 32, dtype=np.uint8),
        np.full((4, 4, 3), 34, dtype=np.uint8),
    )

    assert result.accepted
    assert result.similarity == pytest.approx(0.99999875, abs=1e-6)
