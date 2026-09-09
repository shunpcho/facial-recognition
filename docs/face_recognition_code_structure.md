# 顔認識研究コードの Python 実装骨組み案

## 目的

設計書に基づき、WebFace4M を用いる研究用顔認識コードを最小限の複雑さで実装するための構造案を示す。

本実装では以下を最初の前提に置く。

- データセット: WebFace4M を基準とする
- バックボーン: IR50 を最小限の自前実装で扱う
- 損失: ArcFace を標準ロスとする
- 目標: 顔埋め込み生成と照合評価を両立する
- 実験管理: Lightning + MLflow を利用して再現性と比較性を担保する

---

## 実装の基本方針

顔認識は、入力画像から顔埋め込みを作り、その埋め込み空間で同一人物判定と候補検索を行う流れを考える。

```text
顔画像
  ↓
前処理 (resize / normalize / face crop)
  ↓
IR50 Backbone
  ↓
Embedding
  ↓
ArcFace Loss
  ↓
学習
  ↓
Verification / Identification 評価
```

最初の実装では、AdaFace や ViT などの比較対象を本体に入れず、IR50 + ArcFace の最小構成を中心に据える。バックボーン切り替えの余地は構造として残し、将来の比較研究に備える。

---

## 推奨ディレクトリ構成

このリポジトリでは `src/` レイアウトを採用し、パッケージ名を `facial_recognition` とする。

```text
facial_recognition/
├── pyproject.toml
├── README.md
├── src/
│   └── facial_recognition/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── config.py
│       ├── train.py
│       ├── evaluate.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── metadata.py
│       │   └── transforms.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── ir50.py
│       │   ├── embedding.py
│       │   └── classifier.py
│       ├── losses/
│       │   ├── __init__.py
│       │   └── arcface.py
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── evaluator.py
│       │   └── checkpoint.py
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── verification.py
│       │   └── identification.py
│       └── utils/
│           ├── __init__.py
│           ├── seed.py
│           ├── logging.py
│           └── io.py
├── outputs/
│   └── <run_name>/
│       ├── config.json
│       ├── metrics.json
│       ├── checkpoints/
│       └── artifacts/
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py
│   ├── test_ir50.py
│   ├── test_arcface.py
│   └── test_evaluation.py
└── docs/
    ├── face_recognition_design.md
    └── face_recognition_code_structure.md
```

---

## 設定管理

### `src/facial_recognition/config/config.py`

設定は `dataclass` で定義し、実験条件と実行時の出力先をまとめて管理する。

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    data_dir: Path
    metadata_path: Path
    num_classes: int
    image_size: int = 112
    embedding_dim: int = 512
    scale: float = 64.0
    margin: float = 0.5
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    output_dir: Path = Path("outputs")
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "face_recognition"
    mlflow_run_name: str | None = None
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    use_amp: bool = True
    num_workers: int = 4
```

研究コードの初期実装では、CLI と設定の分離は最小限にし、`train.py` と `evaluate.py` から直接 `TrainConfig` を作って使う方針を採る。

---

## データセット設計

### `src/facial_recognition/data/metadata.py`

WebFace4M を基準にするため、画像と人物 ID を JSON で定義する。`split` を持たせることで、人物単位の train/val/test を安全に分けられる。

```json
[
  { "image_path": "train/00001/0001.jpg", "subject_id": 1, "split": "train" },
  { "image_path": "train/00001/0002.jpg", "subject_id": 1, "split": "train" },
  { "image_path": "val/00002/0001.jpg", "subject_id": 2, "split": "val" },
  { "image_path": "test/00003/0001.jpg", "subject_id": 3, "split": "test" }
]
```

### `src/facial_recognition/data/dataset.py`

- `ImageDataset` は画像と `subject_id` を返す
- フロントエンドで `split` を指定し、人物単位分離を設計する
- `OpenCV` で読み込み、必要に応じて `albumentations` による拡張を追加する

```python
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, records: list[dict[str, object]], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image_path = Path(record["image_path"])  # type: ignore[index]
        subject_id = int(record["subject_id"])  # type: ignore[index]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0
        return image, subject_id
```

---

## モデル設計

### `src/facial_recognition/models/ir50.py`

IR50 は最小限の自前実装として定義する。

- 入力: `B x 3 x H x W`
- 出力: `embedding` と `feature_norm`
- ここではバックボーンの責務だけを持つ

```python
import torch
from torch import nn


class IR50Backbone(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 56 * 56, embedding_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(x)
        x = x.flatten(1)
        embedding = self.fc(x)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding, embedding
```

### `src/facial_recognition/models/embedding.py`

- 入力画像から埋め込みを生成する責務に限定する
- バックボーンの設定や埋め込みの正規化をここで扱う

### `src/facial_recognition/models/classifier.py`

- `embedding` と `labels` を受け取り、識別境界の入力/出力を扱う
- ArcFace の係数や境界の計算をまとめて扱う層として置く

---

## 損失設計

### `src/facial_recognition/losses/arcface.py`

ArcFace のロス本体はここに閉じる。

```python
import torch
from torch import nn


class ArcFaceLoss(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x_norm = nn.functional.normalize(embeddings)
        w_norm = nn.functional.normalize(self.weight)
        cos_theta = nn.functional.linear(x_norm, w_norm)
        target_logits = cos_theta.gather(1, labels.view(-1, 1)).squeeze(1)
        return target_logits.mean()
```

この設計で、損失の詳細は `model` や `engine` の外側に閉じ込み、比較対象と拡張をしやすくする。

---

## 学習ループ

### `src/facial_recognition/engine/trainer.py`

Lightning を使い、学習ループと checkpoint 管理を簡略化する。

```python
import lightning as L
import torch


class FaceRecognitionLightningModel(L.LightningModule):
    def __init__(self, backbone, criterion, lr: float, weight_decay: float):
        super().__init__()
        self.backbone = backbone
        self.criterion = criterion
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embedding, _ = self.backbone(images)
        return embedding

    def training_step(self, batch, batch_idx):
        images, labels = batch
        embeddings, _ = self.backbone(images)
        loss = self.criterion(embeddings, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
```

Lightning が担うもの:

- optimizer / scheduler
- validation loop
- checkpoint 保存
- logging
- run 単位の実行制御

---

## 評価設計

### `src/facial_recognition/engine/evaluator.py`

評価は学習ロジックから独立し、学習後に checkpoint を読み込んで行う。

- Verification: 同一人物/異人物の比較を行い、ROC と FAR/TAR を評価する
- Identification: 候補集合の中で最も近い人物を探し、Rank-1 / Rank-k を評価する

```python
from sklearn.metrics import roc_curve


def compute_verification_metrics(embeddings_a, embeddings_b, labels):
    scores = (embeddings_a * embeddings_b).sum(dim=1)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
```

評価の責務は `verification.py` と `identification.py` に分離し、結果を JSON で保存する。

---

## MLflow 実験管理

### 実験記録の対象

MLflow では以下を記録する。

- hyperparameter
- loss / validation metrics
- checkpoint
- config JSON
- artifact

```python
import mlflow

mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
mlflow.set_experiment(cfg.mlflow_experiment_name)

with mlflow.start_run(run_name=cfg.mlflow_run_name or "ir50_arcface"):
    mlflow.log_params({
        "data_dir": str(cfg.data_dir),
        "num_classes": cfg.num_classes,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "epochs": cfg.epochs,
        "seed": cfg.seed,
    })

    trainer.fit(model, dataloader)
    mlflow.log_artifact(str(cfg.output_dir))
```

これで実験条件の再現性と比較性が保たれる。

---

## 実行入口

### `src/facial_recognition/train.py`

```python
from pathlib import Path

import lightning as L
import mlflow
from torch.utils.data import DataLoader

from facial_recognition.config import TrainConfig
from facial_recognition.data.dataset import ImageDataset
from facial_recognition.data.transforms import build_transform
from facial_recognition.engine.trainer import FaceRecognitionLightningModel
from facial_recognition.losses.arcface import ArcFaceLoss
from facial_recognition.models.ir50 import IR50Backbone


def train(cfg: TrainConfig) -> None:
    transform = build_transform(image_size=cfg.image_size)
    train_records = load_metadata(cfg.metadata_path, cfg.train_split)
    train_dataset = ImageDataset(train_records, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    backbone = IR50Backbone(embedding_dim=cfg.embedding_dim)
    criterion = ArcFaceLoss(
        in_features=cfg.embedding_dim,
        out_features=cfg.num_classes,
        s=cfg.scale,
        m=cfg.margin,
    )

    model = FaceRecognitionLightningModel(
        backbone=backbone,
        criterion=criterion,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    run_dir = cfg.output_dir / (cfg.mlflow_run_name or "ir50_arcface")
    run_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)
    with mlflow.start_run(run_name=cfg.mlflow_run_name or "ir50_arcface"):
        trainer = L.Trainer(
            max_epochs=cfg.epochs,
            default_root_dir=str(run_dir),
            logger=True,
            enable_checkpointing=True,
        )
        trainer.fit(model, train_loader)
        mlflow.log_artifact(str(run_dir))
```

### `src/facial_recognition/evaluate.py`

- 学習済み checkpoint を読む
- validation / test で embedding を生成する
- Verification と Identification を出力する
- 結果を `outputs/<run_name>/metrics.json` に保存する

---

## 実装ルール

### ルール1: バックボーンは特徴抽出器に限定する

- 入力画像から顔の特徴を作る
- embedding と feature_norm を返す
- 識別ロジックや損失計算を持たない

### ルール2: ArcFace は loss 側で閉じる

- `losses/arcface.py` にロス計算を集約する
- `model` や `engine` でロスの実装詳細を直接扱わない

### ルール3: 評価は学習と分離する

- 学習中に評価ロジックを混ぜない
- checkpoint を読み込んだあとで評価を行う

### ルール4: 実験は run 単位で保存する

- 1回の実行ごとに `outputs/<run_name>/` を作る
- `config.json`, `metrics.json`, `checkpoints/` を同じ run 配下に置く

---

## 実験比較の戦略

最初の実験では比較対象を絞る。初期ベースラインは以下の1本を最優先とする。

```text
比較1: WebFace4M + IR50 + ArcFace
```

将来的には、以下のように比較対象を広げる。

```text
比較2: WebFace4M + IR50 + ArcFace + 画像前処理変更
比較3: WebFace4M + IR50 + ArcFace + 別 split 方針
比較4: WebFace4M + 別バックボーン + ArcFace
```

これにより、改善点がバックボーンなのか、前処理なのか、評価設計なのかを切り分けやすくなる。

---

## 依存関係と最小実装

利用する依存は最小限にする。

```text
torch
lightning
mlflow
opencv-python
albumentations  # 必要時のみ
```

目的は、高品質な顔埋め込みを簡潔に生成し、再現性のある研究評価を行うことである。重い依存や過剰な抽象化を避けることが重要である。

---

## まとめ

この骨組みの中心思想は次のとおりである。

- WebFace4M を基準とし、人物単位 split で公平な評価を行う
- IR50 を最小限の自前実装で実現し、研究の差分を損失と評価に集中させる
- ArcFace を中心にし、ロスは `losses/arcface.py` に閉じる
- `model` には埋め込み生成と識別境界の責務を分離する
- Lightning と MLflow により、学習と実験管理を簡潔に保つ
- `train.py` / `evaluate.py` で学習と評価を分離し、`outputs/<run_name>/` に再現可能な結果を残す

この構成により、研究コードとして十分に実用的で、後で比較実験を広げても保守しやすい構造を持つ。
