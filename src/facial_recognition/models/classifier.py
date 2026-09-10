"""Normalized linear classifier for ArcFace training."""

import torch
from torch import nn
from torch.nn import functional


class NormalizedClassifier(nn.Module):
    """Project normalized embeddings onto normalized identity-class weights."""

    def __init__(self, num_classes: int, embedding_dim: int = 512) -> None:
        """Initialize class weights.

        Raises:
            ValueError: If class or embedding dimensions are not positive.
        """
        super().__init__()
        if num_classes <= 0 or embedding_dim <= 0:
            msg = "num_classes and embedding_dim must be positive"
            raise ValueError(msg)
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return cosine similarities for every class."""
        return functional.linear(functional.normalize(embeddings), functional.normalize(self.weight))
