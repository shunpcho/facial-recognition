"""IR-50 backbone producing normalized 512-dimensional face embeddings."""

import torch
from torch import nn
from torch.nn import functional

_INPUT_DIMENSIONS = 4
_IMAGE_SHAPE = (3, 112, 112)


class IRBlock(nn.Module):
    """Pre-activation residual block used by the IR-50 backbone."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        """Initialize a residual block."""
        super().__init__()
        self._shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self._body = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the residual transformation."""
        return self._body(images) + self._shortcut(images)


class IR50(nn.Module):
    """Faithful IR-50-style backbone with fixed 512-D L2-normalized outputs."""

    embedding_dim = 512

    def __init__(self) -> None:
        """Initialize an IR-50 backbone for 112 by 112 RGB faces."""
        super().__init__()
        self._stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        self._stages = nn.Sequential(
            _make_stage(64, 64, block_count=3, stride=2),
            _make_stage(64, 128, block_count=4, stride=2),
            _make_stage(128, 256, block_count=14, stride=2),
            _make_stage(256, 512, block_count=3, stride=2),
        )
        self._output = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.4),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embeddings for RGB 112 by 112 input tensors.

        Args:
            images: Float tensor of shape ``(batch, 3, 112, 112)``.

        Returns:
            Float tensor of shape ``(batch, 512)`` with unit-length rows.

        Raises:
            ValueError: If inputs do not have the IR-50 image shape.
        """
        if images.ndim != _INPUT_DIMENSIONS or images.shape[1:] != _IMAGE_SHAPE:
            msg = "IR50 expects input with shape (batch, 3, 112, 112)"
            raise ValueError(msg)
        features = self._output(self._stages(self._stem(images)))
        return functional.normalize(features, p=2, dim=1)


def _make_stage(in_channels: int, out_channels: int, block_count: int, stride: int) -> nn.Sequential:
    """Build one residual stage with its downsampling in the first block."""
    blocks = [IRBlock(in_channels, out_channels, stride)]
    blocks.extend(IRBlock(out_channels, out_channels, stride=1) for _ in range(block_count - 1))
    return nn.Sequential(*blocks)
