# Facial Recognition

Train a random-initialized IR50 + ArcFace baseline from pre-cropped, aligned
RGB face images. The metadata CSV must include `image_path`, `identity`, and
`split` (`train`, `val`, or `test`) columns. Image paths are relative to the
metadata file.

```bash
uv run train metadata.csv --run-name experiment-001 --accelerator auto
```

Each run writes its resolved configuration, identity mapping, and checkpoints
to `outputs/<run-name>/`. MLflow tracking is local under `mlruns/`.
