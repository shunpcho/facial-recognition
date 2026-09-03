"""CLI for training a face-embedding model with PyTorch."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from facial_recognition.opencv_backend import OpenCVSFaceBackend
from facial_recognition.torch_training import (
    DEFAULT_IMAGE_SIZE,
    prepare_face_training_samples,
    save_torch_checkpoint,
    train_face_embedding_model,
    TrainingConfig,
)


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Train a face recognition embedding model with PyTorch.")
    parser.add_argument("dataset_dir", type=Path, help="Directory of identity subdirectories with face images.")
    parser.add_argument("checkpoint_path", type=Path, help="Output path for the trained PyTorch checkpoint.")
    parser.add_argument("--detector-model", type=Path, required=True, help="Absolute path to the YuNet ONNX model.")
    parser.add_argument("--aligner-model", type=Path, required=True, help="Absolute path to the SFace ONNX model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding vector size.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE[0],
        help="Square aligned face size used for training.",
    )
    parser.add_argument("--threshold", type=float, default=0.363, help="Default cosine threshold to store.")
    parser.add_argument("--device", default="cpu", help='Torch device, for example "cpu", "cuda", or "auto".')
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed for reproducible training.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the training CLI."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    image_size = (args.image_size, args.image_size)
    alignment_backend = OpenCVSFaceBackend(
        detector_model_path=args.detector_model,
        recognizer_model_path=args.aligner_model,
    )
    samples, class_names = prepare_face_training_samples(
        dataset_dir=args.dataset_dir,
        alignment_backend=alignment_backend,
        image_size=image_size,
    )
    embedder, metrics = train_face_embedding_model(
        samples=samples,
        num_classes=len(class_names),
        config=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            embedding_dim=args.embedding_dim,
            image_size=image_size,
            random_seed=args.random_seed,
            device=args.device,
        ),
    )
    save_torch_checkpoint(
        checkpoint_path=args.checkpoint_path,
        model=embedder,
        class_names=class_names,
        image_size=image_size,
        threshold=args.threshold,
    )
    final_metrics = metrics[-1]
    print(
        "trained checkpoint="
        f"{args.checkpoint_path} classes={len(class_names)} "
        f"epoch={final_metrics.epoch} loss={final_metrics.loss:.4f} accuracy={final_metrics.accuracy:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
