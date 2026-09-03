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
from facial_recognition.torch_backend import TorchEmbeddingBackend
from facial_recognition.torch_training import (
    FaceEmbeddingNet,
    load_torch_checkpoint,
    prepare_face_training_samples,
    PreparedFaceSample,
    save_torch_checkpoint,
    train_face_embedding_model,
    TrainingConfig,
)

__all__ = [
    "ComparisonResult",
    "FaceDetection",
    "FaceEmbeddingNet",
    "FaceGallery",
    "FaceRecognitionPipeline",
    "MatchResult",
    "OpenCVSFaceBackend",
    "PreparedFaceSample",
    "RecognitionBackend",
    "TorchEmbeddingBackend",
    "TrainingConfig",
    "cosine_similarity",
    "load_torch_checkpoint",
    "prepare_face_training_samples",
    "preprocess_image",
    "save_torch_checkpoint",
    "train_face_embedding_model",
]
