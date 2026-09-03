"""OpenCV-based face detection and embedding extraction backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from facial_recognition.core import Embedding, FaceDetection, ImageArray


@dataclass(frozen=True, slots=True)
class OpenCVSFaceBackend:
    """OpenCV YuNet + SFace backend."""

    detector_model_path: Path
    recognizer_model_path: Path
    input_size: tuple[int, int] = (640, 640)
    score_threshold: float = 0.9
    nms_threshold: float = 0.3
    top_k: int = 5000

    def __post_init__(self) -> None:
        object.__setattr__(self, "_detector", self._create_detector())
        object.__setattr__(self, "_recognizer", self._create_recognizer())

    def _create_detector(self) -> cv2.FaceDetectorYN:
        model_path = self.detector_model_path.expanduser().resolve()
        return cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            self.input_size,
            self.score_threshold,
            self.nms_threshold,
            self.top_k,
        )

    def _create_recognizer(self) -> cv2.FaceRecognizerSF:
        model_path = self.recognizer_model_path.expanduser().resolve()
        return cv2.FaceRecognizerSF.create(str(model_path), "")

    def detect(self, image: ImageArray) -> list[FaceDetection]:
        """Detects faces and landmarks with YuNet."""

        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image)
        if faces is None:
            return []

        detections: list[FaceDetection] = []
        for row in faces.astype(np.float32, copy=False):
            x, y, box_width, box_height = row[:4]
            landmarks = tuple((float(row[index]), float(row[index + 1])) for index in range(4, 14, 2))
            detections.append(
                FaceDetection(
                    bounding_box=(int(round(x)), int(round(y)), int(round(box_width)), int(round(box_height))),
                    confidence=float(row[14]),
                    landmarks=landmarks,
                    raw_detection=np.ascontiguousarray(row),
                )
            )
        return detections

    def align(self, image: ImageArray, detection: FaceDetection) -> ImageArray:
        """Aligns a face crop using SFace landmarks."""

        if detection.raw_detection is None:
            msg = "OpenCV alignment requires the raw detector output."
            raise ValueError(msg)
        aligned_face = self._recognizer.alignCrop(image, detection.raw_detection)
        return np.ascontiguousarray(aligned_face)

    def extract(self, aligned_face: ImageArray) -> Embedding:
        """Extracts an SFace embedding."""

        embedding = self._recognizer.feature(aligned_face)
        return np.asarray(embedding, dtype=np.float32).reshape(-1)
