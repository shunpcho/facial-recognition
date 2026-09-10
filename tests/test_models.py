"""Tests for the embedding backbone and ArcFace classification boundary."""

import torch

from facial_recognition.losses.arcface import ArcFaceLoss
from facial_recognition.models.classifier import NormalizedClassifier
from facial_recognition.models.ir50 import IR50


def test_ir50_returns_unit_length_embeddings() -> None:
    model = IR50().eval()

    with torch.no_grad():
        embeddings = model(torch.randn(2, 3, 112, 112))

    assert embeddings.shape == (2, 512)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(2), atol=1e-5)


def test_arcface_replaces_only_target_logit() -> None:
    cosine_logits = torch.tensor([[0.8, 0.1], [0.2, 0.7]])
    labels = torch.tensor([0, 1])

    logits = ArcFaceLoss(scale=1.0, margin=0.5)(cosine_logits, labels)

    assert logits[0, 0] < cosine_logits[0, 0]
    assert logits[1, 1] < cosine_logits[1, 1]
    assert logits[0, 1] == cosine_logits[0, 1]
    assert logits[1, 0] == cosine_logits[1, 0]


def test_normalized_classifier_returns_one_logit_per_class() -> None:
    classifier = NormalizedClassifier(num_classes=3)

    logits = classifier(torch.randn(2, 512))

    assert logits.shape == (2, 3)
