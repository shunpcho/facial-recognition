"""ArcFace angular-margin logits."""

import math

import torch
from torch import nn

_LOGIT_DIMENSIONS = 2


class ArcFaceLoss(nn.Module):
    """Apply an additive angular margin to target-class cosine similarities."""

    def __init__(self, scale: float = 64.0, margin: float = 0.5) -> None:
        """Initialize the ArcFace margin transform.

        Raises:
            ValueError: If scale or margin is outside a valid range.
        """
        super().__init__()
        if scale <= 0:
            msg = "scale must be positive"
            raise ValueError(msg)
        if not 0 < margin < math.pi / 2:
            msg = "margin must be between zero and pi/2"
            raise ValueError(msg)
        self._scale = scale
        self._cos_margin = math.cos(margin)
        self._sin_margin = math.sin(margin)

    def forward(self, cosine_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply the margin to target logits and multiply all logits by scale.

        Raises:
            ValueError: If labels do not define one valid class per input row.
        """
        if cosine_logits.ndim != _LOGIT_DIMENSIONS or labels.ndim != 1 or cosine_logits.shape[0] != labels.shape[0]:
            msg = "cosine_logits must be (batch, classes) and labels must be (batch,)"
            raise ValueError(msg)
        if labels.numel() and (labels.min() < 0 or labels.max() >= cosine_logits.shape[1]):
            msg = "labels must be valid class indices"
            raise ValueError(msg)

        target_cosine = cosine_logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        target_sine = torch.sqrt((1.0 - target_cosine.square()).clamp_min(0.0))
        target_with_margin = target_cosine * self._cos_margin - target_sine * self._sin_margin
        logits = cosine_logits.clone()
        logits.scatter_(1, labels.unsqueeze(1), target_with_margin.unsqueeze(1))
        return logits * self._scale
