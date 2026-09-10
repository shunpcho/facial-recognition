"""CSV metadata parsing and deterministic identity-label normalization."""

import csv
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

DatasetSplit = Literal["train", "val", "test"]
_VALID_SPLITS = frozenset(("train", "val", "test"))


@dataclass(frozen=True, slots=True)
class FaceRecord:
    """One valid face-image record from the metadata file."""

    image_path: Path
    identity: str
    split: DatasetSplit


def load_metadata(metadata_path: Path) -> list[FaceRecord]:
    """Load valid records from a CSV file, logging warnings for skipped rows.

    The CSV must provide ``image_path``, ``identity``, and ``split`` columns.

    Args:
        metadata_path: Location of the metadata CSV file.

    Returns:
        Valid records in input order.

    Raises:
        FileNotFoundError: If the metadata file is absent.
        ValueError: If the metadata header is invalid or no valid records remain.
    """
    with metadata_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        _validate_header(reader.fieldnames, metadata_path)
        records = [
            record
            for row_number, row in enumerate(reader, start=2)
            if (record := _parse_record(row, row_number, metadata_path.parent)) is not None
        ]

    if not records:
        msg = f"No valid records remain in metadata: {metadata_path}"
        raise ValueError(msg)
    return records


def records_for_split(records: Iterable[FaceRecord], split: DatasetSplit) -> list[FaceRecord]:
    """Return records for one split or fail if skipping left it unusable.

    Args:
        records: Parsed metadata records.
        split: Requested dataset split.

    Raises:
        ValueError: If the requested split has no valid records.
    """
    selected_records = [record for record in records if record.split == split]
    if not selected_records:
        msg = f"No valid records remain for split: {split}"
        raise ValueError(msg)
    return selected_records


def build_identity_mapping(records: Iterable[FaceRecord]) -> dict[str, int]:
    """Map training identities to sorted contiguous labels.

    Args:
        records: Training records whose identities define classifier classes.

    Returns:
        Mapping from original identity strings to contiguous class indices.

    Raises:
        ValueError: If no identities are present.
    """
    identities = sorted({record.identity for record in records})
    if not identities:
        msg = "Cannot build an identity mapping without training records"
        raise ValueError(msg)
    return {identity: index for index, identity in enumerate(identities)}


def _validate_header(fieldnames: Sequence[str] | None, metadata_path: Path) -> None:
    """Require the stable metadata columns before interpreting any rows."""
    required_columns = {"image_path", "identity", "split"}
    actual_columns = set(fieldnames or ())
    missing_columns = sorted(required_columns - actual_columns)
    if missing_columns:
        formatted_columns = ", ".join(missing_columns)
        msg = f"Metadata is missing required columns ({formatted_columns}): {metadata_path}"
        raise ValueError(msg)


def _parse_record(
    row: dict[str | None, str | list[str] | None],
    row_number: int,
    metadata_directory: Path,
) -> FaceRecord | None:
    """Validate one metadata row and return ``None`` after a warning if invalid."""
    image_path_value = _string_value(row.get("image_path"))
    identity = _string_value(row.get("identity"))
    split = _string_value(row.get("split"))
    if not image_path_value or not identity or not split:
        logger.warning("Skipping metadata row %d: image_path, identity, and split are required", row_number)
        return None
    if split not in _VALID_SPLITS:
        logger.warning("Skipping metadata row %d: unsupported split %r", row_number, split)
        return None

    image_path = metadata_directory / image_path_value
    if not image_path.is_file():
        logger.warning("Skipping metadata row %d: image does not exist: %s", row_number, image_path)
        return None
    return FaceRecord(image_path=image_path, identity=identity, split=split)


def _string_value(value: str | list[str] | None) -> str:
    """Normalize CSV scalar fields while treating malformed list values as empty."""
    return value.strip() if isinstance(value, str) else ""
