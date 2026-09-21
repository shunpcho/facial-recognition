"""Tests for CSV metadata validation and deterministic labels."""

import logging
from pathlib import Path

import pytest
from PIL import Image

from facial_recognition.datasets.metadata import build_identity_mapping, load_metadata, records_for_split


def test_load_metadata_skips_invalid_rows_and_normalizes_relative_paths(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    image_path = tmp_path / "face.png"
    Image.new("RGB", (8, 8)).save(image_path)
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "image_path,identity,split\nface.png,person-b,train\nmissing.png,person-a,train\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        records = load_metadata(metadata_path)

    assert records[0].image_path == image_path
    assert records[0].identity == "person-b"
    assert "image does not exist" in caplog.text


def test_identity_mapping_is_sorted_and_contiguous(tmp_path: Path) -> None:
    image_path = tmp_path / "face.png"
    Image.new("RGB", (8, 8)).save(image_path)
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "image_path,identity,split\nface.png,zulu,train\nface.png,alpha,train\n",
        encoding="utf-8",
    )

    mapping = build_identity_mapping(records_for_split(load_metadata(metadata_path), "train"))

    assert mapping == {"alpha": 0, "zulu": 1}


def test_records_for_split_rejects_unusable_split(tmp_path: Path) -> None:
    image_path = tmp_path / "face.png"
    Image.new("RGB", (8, 8)).save(image_path)
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text("image_path,identity,split\nface.png,alpha,train\n", encoding="utf-8")

    with pytest.raises(ValueError, match="val"):
        records_for_split(load_metadata(metadata_path), "val")
