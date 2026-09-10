"""Torch dataset implementation for validated face-image records."""

from collections.abc import Callable, Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from facial_recognition.datasets.metadata import FaceRecord

ImageTransform = Callable[[Image.Image], torch.Tensor]


class FaceDataset(Dataset[tuple[torch.Tensor, int]]):
    """Return transformed face images paired with normalized identity labels."""

    def __init__(
        self,
        records: Sequence[FaceRecord],
        identity_mapping: Mapping[str, int],
        transform: ImageTransform,
    ) -> None:
        """Create a dataset from already validated records.

        Args:
            records: Records for one usable dataset split.
            identity_mapping: Persistent identity-to-class mapping from training data.
            transform: Image preprocessing callable.

        Raises:
            ValueError: If a record identity is absent from the training mapping.
        """
        unmapped_identities = {record.identity for record in records} - identity_mapping.keys()
        if unmapped_identities:
            formatted_identities = ", ".join(sorted(unmapped_identities))
            msg = f"Records include identities absent from the training mapping: {formatted_identities}"
            raise ValueError(msg)
        self._records = list(records)
        self._identity_mapping = dict(identity_mapping)
        self._transform = transform

    def __len__(self) -> int:
        """Return the number of validated records."""
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load and transform one pre-aligned face image."""
        record = self._records[index]
        with Image.open(record.image_path) as image:
            transformed_image = self._transform(image)
        return transformed_image, self._identity_mapping[record.identity]
