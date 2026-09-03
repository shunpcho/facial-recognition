"""OpenCV-based face detection and embedding extraction backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

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
    _detector: object = field(init=False, repr=False)
    _recognizer: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_detector", self._create_detector())
        object.__setattr__(self, "_recognizer", self._create_recognizer())

    def _create_detector(self) -> object:
        model_path = self.detector_model_path.expanduser().resolve()
        return cast(
            "object",
            cv2.FaceDetectorYN.create(
                str(model_path),
                "",
                self.input_size,
                self.score_threshold,
                self.nms_threshold,
                self.top_k,
            ),
        )

    def _create_recognizer(self) -> object:
        model_path = self.recognizer_model_path.expanduser().resolve()
        return cast("object", cv2.FaceRecognizerSF.create(str(model_path), ""))

    def detect(self, image: ImageArray) -> list[FaceDetection]:
        """Detects faces and landmarks with YuNet."""
        height, width = image.shape[:2]
        detector = cast("cv2.FaceDetectorYN", self._detector)
        detector.setInputSize((width, height))
        _, faces = cast("tuple[object, npt.NDArray[np.float32] | None]", detector.detect(image))
        if faces is None:
            return []

        detections: list[FaceDetection] = []
        for row in faces.astype(np.float32, copy=False):
            x = float(row[0])
            y = float(row[1])
            box_width = float(row[2])
            box_height = float(row[3])
            landmarks = tuple((float(row[index]), float(row[index + 1])) for index in range(4, 14, 2))
            detections.append(
                FaceDetection(
                    bounding_box=(round(x), round(y), round(box_width), round(box_height)),
                    confidence=float(row[14]),
                    landmarks=landmarks,
                    raw_detection=np.ascontiguousarray(row),
                )
            )
        return detections

    def align(self, image: ImageArray, detection: FaceDetection) -> ImageArray:
        """Aligns a face crop using SFace landmarks.

        Raises:
            ValueError: If the detection lacks raw detector metadata.
        """
        if detection.raw_detection is None:
            msg = "OpenCV alignment requires the raw detector output."
            raise ValueError(msg)
        recognizer = cast("cv2.FaceRecognizerSF", self._recognizer)
        aligned_face = cast("npt.NDArray[np.uint8]", recognizer.alignCrop(image, detection.raw_detection))
        return np.ascontiguousarray(aligned_face)

    def extract(self, aligned_face: ImageArray) -> Embedding:
        """Extracts an SFace embedding."""
        recognizer = cast("cv2.FaceRecognizerSF", self._recognizer)
        embedding = cast("npt.NDArray[np.float32]", recognizer.feature(aligned_face))
        return np.asarray(embedding, dtype=np.float32).reshape(-1)
