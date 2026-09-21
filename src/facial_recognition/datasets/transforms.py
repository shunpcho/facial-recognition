"""Image transforms with explicit train and validation behavior."""

import random
from typing import Literal

import numpy as np
import torch
from PIL import Image

TransformMode = Literal["train", "validation"]

_NORMALIZATION_MEAN = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float32).view(3, 1, 1)
_NORMALIZATION_STD = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float32).view(3, 1, 1)
_HORIZONTAL_FLIP_PROBABILITY = 0.5


class FaceTransform:
    """Resize RGB face images and normalize them to the ``[-1, 1]`` range."""

    def __init__(self, mode: TransformMode, image_size: int = 112) -> None:
        """Initialize a transform with its explicit augmentation mode.

        Args:
            mode: ``train`` enables horizontal flipping; ``validation`` is deterministic.
            image_size: Target square image dimension.

        Raises:
            ValueError: If the image size is not positive.
        """
        if image_size <= 0:
            msg = "image_size must be positive"
            raise ValueError(msg)
        self._mode = mode
        self._image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Transform one image into a normalized CHW float32 tensor."""
        rgb_image = image.convert("RGB").resize((self._image_size, self._image_size))
        if self._mode == "train" and random.random() < _HORIZONTAL_FLIP_PROBABILITY:
            rgb_image = rgb_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        array = np.array(rgb_image, dtype=np.float32, copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).div(255.0)
        return (tensor - _NORMALIZATION_MEAN) / _NORMALIZATION_STD


def build_transform(mode: TransformMode, image_size: int = 112) -> FaceTransform:
    """Build an explicit training or validation transform."""
    return FaceTransform(mode=mode, image_size=image_size)
