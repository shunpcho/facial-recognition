"""Pairwise verification metrics for normalized face embeddings."""

from dataclasses import dataclass
from typing import cast

import torch
from torch.nn import functional

_EMBEDDING_DIMENSIONS = 2


@dataclass(frozen=True, slots=True)
class VerificationMetrics:
    """Summary statistics for a declared verification-pair protocol."""

    roc_auc: float
    eer: float
    tar_at_far_1e3: float
    tar_at_far_1e4: float


def evaluate_verification(
    first_embeddings: torch.Tensor, second_embeddings: torch.Tensor, is_same_identity: torch.Tensor
) -> VerificationMetrics:
    """Compute ROC-AUC, EER, and TAR at fixed FAR operating points.

    Raises:
        ValueError: If inputs do not describe at least one genuine and impostor pair.
    """
    if first_embeddings.shape != second_embeddings.shape or first_embeddings.ndim != _EMBEDDING_DIMENSIONS:
        msg = "embedding tensors must have matching shape (pairs, dimensions)"
        raise ValueError(msg)
    if is_same_identity.shape != (first_embeddings.shape[0],):
        msg = "is_same_identity must have shape (pairs,)"
        raise ValueError(msg)
    scores = functional.cosine_similarity(first_embeddings, second_embeddings)
    positives = is_same_identity.to(dtype=torch.bool)
    if not positives.any() or positives.all():
        msg = "verification requires both genuine and impostor pairs"
        raise ValueError(msg)

    thresholds = cast("torch.Tensor", torch.unique(scores).sort().values)
    tar = torch.stack([(scores[positives] >= threshold).float().mean() for threshold in thresholds])
    far = torch.stack([(scores[~positives] >= threshold).float().mean() for threshold in thresholds])
    false_rejection_rate = 1.0 - tar
    eer_index = torch.argmin((far - false_rejection_rate).abs())
    auc = torch.trapezoid(tar, far).abs()
    return VerificationMetrics(
        roc_auc=float(auc),
        eer=float((far[eer_index] + false_rejection_rate[eer_index]) / 2),
        tar_at_far_1e3=_tar_at_far(tar, far, 1e-3),
        tar_at_far_1e4=_tar_at_far(tar, far, 1e-4),
    )


def _tar_at_far(tar: torch.Tensor, far: torch.Tensor, maximum_far: float) -> float:
    """Return the best TAR among thresholds satisfying an FAR constraint."""
    valid = tar[far <= maximum_far]
    return float(valid.max()) if valid.numel() else 0.0
