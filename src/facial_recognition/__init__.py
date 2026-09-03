"""Public API for the facial recognition pipeline."""

from facial_recognition.core import (
    ComparisonResult,
    cosine_similarity,
    FaceDetection,
    FaceGallery,
    FaceRecognitionPipeline,
    MatchResult,
    preprocess_image,
    RecognitionBackend,
)
from facial_recognition.opencv_backend import OpenCVSFaceBackend

__all__ = [
    "ComparisonResult",
    "FaceDetection",
    "FaceGallery",
    "FaceRecognitionPipeline",
    "MatchResult",
    "OpenCVSFaceBackend",
    "RecognitionBackend",
    "cosine_similarity",
    "preprocess_image",
]
