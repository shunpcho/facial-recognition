# Facial Recognition

OpenCV の YuNet/SFace を使った顔認識パイプラインを提供します。

## 機能

- 顔検出
- 5 点ランドマークによる整列
- 画像前処理
- 埋め込み特徴抽出
- コサイン類似度による照合
- しきい値ベースの本人判定

## セットアップ

```bash
uv sync --all-groups
```

## モデル

以下の ONNX モデルを用意し、コードから絶対パスで指定してください。

- YuNet face detector
- SFace face recognizer

OpenCV Zoo の公開モデルをそのまま使えます。

## 使用例

```python
from pathlib import Path

import cv2

from facial_recognition import FaceRecognitionPipeline, OpenCVSFaceBackend

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

## 開発コマンド

```bash
uv run ruff format .
uv run ruff check .
uv run pyright src tests
uv run pytest .
```
