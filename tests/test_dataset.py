"""Tests for transformed face datasets."""

from pathlib import Path

import torch
from PIL import Image

from facial_recognition.datasets.face_dataset import FaceDataset
from facial_recognition.datasets.metadata import FaceRecord
from facial_recognition.datasets.transforms import build_transform


def test_face_dataset_returns_normalized_image_and_label(tmp_path: Path) -> None:
    image_path = tmp_path / "face.png"
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(image_path)
    records = [FaceRecord(image_path=image_path, identity="alpha", split="train")]

    dataset = FaceDataset(records, {"alpha": 0}, build_transform("validation", image_size=8))
    image, label = dataset[0]

    assert image.shape == (3, 8, 8)
    assert image.dtype == torch.float32
    assert label == 0
