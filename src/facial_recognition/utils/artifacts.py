"""Filesystem layout and atomic artifact writes for experiment runs."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Filesystem locations belonging to one named experiment run."""

    root: Path
    checkpoints: Path
    artifacts: Path

    @classmethod
    def create(cls, output_dir: Path, run_name: str) -> RunPaths:
        """Create a deterministic run directory and its required children."""
        root = output_dir / run_name
        checkpoints = root / "checkpoints"
        artifacts = root / "artifacts"
        checkpoints.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)
        return cls(root=root, checkpoints=checkpoints, artifacts=artifacts)

    def write_config(self, config: Mapping[str, str | int | float]) -> Path:
        """Atomically persist the fully resolved run configuration."""
        return _write_json(self.root / "config.json", config)


def _write_json(path: Path, payload: Mapping[str, str | int | float]) -> Path:
    """Write JSON atomically so interrupted processes never leave partial metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        json.dump(payload, temporary_file, indent=2, sort_keys=True)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)

    temporary_path.replace(path)
    return path
