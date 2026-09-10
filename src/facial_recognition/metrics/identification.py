"""Gallery/probe identification metrics for normalized face embeddings."""

from dataclasses import dataclass

import torch
from torch.nn import functional

_EMBEDDING_DIMENSIONS = 2


@dataclass(frozen=True, slots=True)
class IdentificationMetrics:
    """Closed-set gallery/probe search metrics."""

    rank_1: float
    rank_5: float
    mean_average_precision: float


def evaluate_identification(
    gallery_embeddings: torch.Tensor,
    gallery_identities: torch.Tensor,
    probe_embeddings: torch.Tensor,
    probe_identities: torch.Tensor,
) -> IdentificationMetrics:
    """Calculate CMC rank-1/rank-5 and mAP for a declared gallery/probe split."""
    if gallery_embeddings.ndim != _EMBEDDING_DIMENSIONS or probe_embeddings.ndim != _EMBEDDING_DIMENSIONS:
        msg = "embedding tensors must have shape (samples, dimensions)"
        raise ValueError(msg)
    if gallery_embeddings.shape[1] != probe_embeddings.shape[1]:
        msg = "gallery and probe embedding dimensions must match"
        raise ValueError(msg)
    if gallery_identities.shape != (gallery_embeddings.shape[0],) or probe_identities.shape != (
        probe_embeddings.shape[0],
    ):
        msg = "identity tensors must contain one label per embedding"
        raise ValueError(msg)
    if not gallery_embeddings.shape[0] or not probe_embeddings.shape[0]:
        msg = "identification requires non-empty gallery and probe sets"
        raise ValueError(msg)

    scores = functional.normalize(probe_embeddings) @ functional.normalize(gallery_embeddings).T
    ranking = scores.argsort(dim=1, descending=True)
    ranked_identities = gallery_identities[ranking]
    matches = ranked_identities == probe_identities.unsqueeze(1)
    if not matches.any(dim=1).all():
        msg = "every probe identity must occur in the gallery"
        raise ValueError(msg)
    rank_1 = matches[:, 0].float().mean()
    rank_5 = matches[:, : min(5, matches.shape[1])].any(dim=1).float().mean()
    positions = torch.arange(1, matches.shape[1] + 1, device=matches.device)
    precision = matches.cumsum(dim=1) / positions
    average_precision = (precision * matches).sum(dim=1) / matches.sum(dim=1)
    return IdentificationMetrics(float(rank_1), float(rank_5), float(average_precision.mean()))
