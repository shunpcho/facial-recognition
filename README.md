# Facial Recognition

OpenCV の YuNet/SFace と PyTorch を使った顔認識パイプラインを提供します。

## 機能

- 顔検出
- 5 点ランドマークによる整列
- 画像前処理
- 埋め込み特徴抽出
- コサイン類似度による照合
- しきい値ベースの本人判定
- PyTorch による学習済み埋め込みモデル作成

## セットアップ

```bash
uv sync --all-groups
```

## モデル

以下の ONNX モデルを用意し、コードから絶対パスで指定してください。

- YuNet face detector
- SFace face recognizer

OpenCV Zoo の公開モデルをそのまま使えます。

## PyTorch 学習用データセット

学習データは以下のように人物ごとのディレクトリで配置します。

```text
/absolute/path/to/dataset/
├── alice/
│   ├── 001.jpg
│   └── 002.jpg
└── bob/
    ├── 001.jpg
    └── 002.jpg
```

少なくとも 2 人分のディレクトリが必要です。

## PyTorch 学習

```bash
uv run train \
  /absolute/path/to/dataset \
  /absolute/path/to/checkpoints/face_embedder.pt \
  --detector-model /absolute/path/to/face_detection_yunet.onnx \
  --aligner-model /absolute/path/to/face_recognition_sface.onnx \
  --epochs 20 \
  --batch-size 16 \
  --embedding-dim 128 \
  --device auto
```

学習時は画像を顔検出・整列してから PyTorch の埋め込みモデルを学習し、重み・クラス一覧・入力サイズ・既定しきい値を checkpoint に保存します。

## 使用例

```python
from pathlib import Path

import cv2

from facial_recognition import FaceRecognitionPipeline, OpenCVSFaceBackend, TorchEmbeddingBackend

backend = OpenCVSFaceBackend(
    detector_model_path=Path("/absolute/path/to/face_detection_yunet.onnx"),
    recognizer_model_path=Path("/absolute/path/to/face_recognition_sface.onnx"),
)
pipeline = FaceRecognitionPipeline(backend=backend, threshold=0.363)

alice_image = cv2.imread("/absolute/path/to/alice.jpg")
query_image = cv2.imread("/absolute/path/to/query.jpg")

pipeline.enroll("alice", alice_image)
result = pipeline.recognize(query_image)

print(result)
```

## 学習済み PyTorch モデルで推論

```python
from pathlib import Path

import cv2

from facial_recognition import FaceRecognitionPipeline, OpenCVSFaceBackend, TorchEmbeddingBackend

alignment_backend = OpenCVSFaceBackend(
    detector_model_path=Path("/absolute/path/to/face_detection_yunet.onnx"),
    recognizer_model_path=Path("/absolute/path/to/face_recognition_sface.onnx"),
)
backend = TorchEmbeddingBackend.from_checkpoint(
    checkpoint_path=Path("/absolute/path/to/checkpoints/face_embedder.pt"),
    alignment_backend=alignment_backend,
    device="cpu",
)
pipeline = FaceRecognitionPipeline(backend=backend, threshold=backend.threshold)

alice_image = cv2.imread("/absolute/path/to/alice.jpg")
query_image = cv2.imread("/absolute/path/to/query.jpg")

pipeline.enroll("alice", alice_image)
result = pipeline.recognize(query_image)

print(result)
```

## 開発コマンド

```bash
uv run ruff format .
uv run ruff check .
uv run pyright src tests
uv run pytest .
```
