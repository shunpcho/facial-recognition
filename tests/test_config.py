"""Tests for typed experiment configuration."""

from pathlib import Path

import pytest

from facial_recognition.config.config import TrainConfig


def test_config_serializes_paths() -> None:
    config = TrainConfig(metadata_path=Path("data/metadata.csv"))

    assert config.as_serializable_dict()["metadata_path"] == "data/metadata.csv"


def test_config_rejects_invalid_training_values() -> None:
    with pytest.raises(ValueError, match="run_name"):
        TrainConfig(metadata_path=Path("metadata.csv"), run_name="")
    with pytest.raises(ValueError, match="epochs"):
        TrainConfig(metadata_path=Path("metadata.csv"), epochs=0)
    with pytest.raises(ValueError, match="batch_size"):
        TrainConfig(metadata_path=Path("metadata.csv"), batch_size=0)
    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig(metadata_path=Path("metadata.csv"), learning_rate=0.0)
    with pytest.raises(ValueError, match="weight_decay"):
        TrainConfig(metadata_path=Path("metadata.csv"), weight_decay=-0.1)
