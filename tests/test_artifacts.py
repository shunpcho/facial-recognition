"""Tests for deterministic, atomic run artifact storage."""

import json
from pathlib import Path

from facial_recognition.utils.artifacts import RunPaths


def test_run_paths_write_config(tmp_path: Path) -> None:
    run_paths = RunPaths.create(tmp_path / "outputs", "experiment-001")

    config_path = run_paths.write_config({"seed": 42, "metadata_path": "metadata.csv"})

    assert run_paths.checkpoints.is_dir()
    assert run_paths.artifacts.is_dir()
    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "metadata_path": "metadata.csv",
        "seed": 42,
    }
