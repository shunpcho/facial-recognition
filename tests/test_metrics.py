"""Tests for verification and identification metric contracts."""

import torch

from facial_recognition.metrics.identification import evaluate_identification
from facial_recognition.metrics.verification import evaluate_verification


def test_verification_reports_perfect_separation() -> None:
    metrics = evaluate_verification(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([True, False]),
    )
    assert metrics.roc_auc == 1.0
    assert metrics.tar_at_far_1e3 == 1.0


def test_identification_reports_perfect_ranking() -> None:
    metrics = evaluate_identification(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([0, 1]),
        torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
        torch.tensor([0, 1]),
    )
    assert metrics.rank_1 == 1.0
    assert metrics.rank_5 == 1.0
    assert metrics.mean_average_precision == 1.0
