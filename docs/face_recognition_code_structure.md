# 顔認識研究コードの Python 骨組み案

## 目的

設計書をもとに、研究用顔認識コードを最小限の複雑さで実装するための構造案を示す。

### 処理の要点

顔認識は、画像前処理から識別性能評価までを段階的に分けて扱う。

```text
顔画像 → 前処理 → Backbone → Embedding → Partial FC → ArcFace / AdaFace → 学習 → 評価
```

前処理で入力を揃え、Backbone で特徴を抽出し、埋め込みと識別境界を作り、最後に照合や検索性能を確認する。

---

## 推奨ディレクトリ構成

```text
face_recognition/
├── train.py
├── evaluate.py
├── config/
│   ├── __init__.py
│   └── config.py
├── datasets/
│   ├── __init__.py
│   ├── face_dataset.py
│   ├── transforms.py
│   └── sampler.py
├── models/
│   ├── __init__.py
│   ├── backbone.py
│   ├── vit.py
│   ├── ir.py
│   ├── embedding.py
│   └── classifier.py
├── losses/
│   ├── __init__.py
│   ├── arcface.py
│   ├── adaface.py
│   └── margin.py
├── engine/
│   ├── __init__.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── checkpoint.py
│   └── distributed.py
├── utils/
│   ├── __init__.py
│   ├── seed.py
│   ├── logger.py
│   └── metrics.py
└── tests/
    ├── __init__.py
    ├── test_dataset.py
    ├── test_backbone.py
    ├── test_loss.py
    └── test_training_loop.py
```

---

## 設定管理

### `config/config.py`

- CLI と実験条件を `cyclopts` の設定オブジェクトにまとめる
- `dataclass` を中心に定義し、型の情報とヘルプ表示を保持する
- `argparse` より簡潔に実験条件を切り替えられる

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cyclopts import App

app = App()


@dataclass
class TrainConfig:
    train_data: str
    num_classes: int = 10000
    image_size: int = 112
    model: Literal["vit", "ir50", "ir100"] = "vit"
    embedding_dim: int = 512
    margin: Literal["arcface", "adaface"] = "arcface"
    scale: float = 64.0
    margin_value: float = 0.5
    adaface_h: float = 0.2
    sample_rate: float = 0.1
    epochs: int = 20
    batch_size: int = 128
    optimizer: Literal["sgd", "adamw"] = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    output_dir: Path = Path("outputs")
    seed: int = 42
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "face_recognition"
    mlflow_run_name: str | None = None
```

```python
@app.default
def train(cfg: TrainConfig):
    print(cfg)
```

```bash
python train.py --train-data data/train --model vit --margin arcface --epochs 20
```

`argparse` では `ArgumentParser` と `add_argument` を手で定義する必要があるが、`cyclopts` では設定そのものが CLI と一体になる。型情報とヘルプが自然に残るため、研究コードに向いている。

---

## 学習ループ

### `engine/trainer.py`

```python
import lightning as L
import torch


class FaceRecognitionModel(L.LightningModule):
    def __init__(self, model, classifier, cfg):
        super().__init__()
        self.model = model
        self.classifier = classifier
        self.cfg = cfg

    def forward(self, images):
        embedding, norm = self.model(images)
        return embedding, norm

    def training_step(self, batch, batch_idx):
        images, labels = batch
        embedding, norm = self.model(images)
        logits = self.classifier(embedding, labels, norm=norm)
        loss = logits.loss if hasattr(logits, "loss") else logits
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
```

Lightning は以下を自動で管理する。

- optimizer / scheduler
- validation loop
- checkpoint 保存
- logging
- distributed training

---

## 実験記録

学習の再現性と比較性を保つため、MLflow に実験条件と結果を保存する。

記録するもの:

- 実験名と run 名
- 学習設定（model, margin, lr, batch_size, epochs など）
- loss / validation metrics
- checkpoint と artifact
- 実行時の設定 JSON

```python
import mlflow

mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
mlflow.set_experiment(cfg.mlflow_experiment_name)

with mlflow.start_run(run_name=cfg.mlflow_run_name or cfg.model):
    mlflow.log_params({
        "model": cfg.model,
        "margin": cfg.margin,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "seed": cfg.seed,
    })

    trainer = L.Trainer(...)
    trainer.fit(pl_model, loader)
    mlflow.log_artifact(str(cfg.output_dir))
```

これにより、実験条件の再現と結果比較がしやすくなる。

---

## 実行入口

`train.py` を起点にして、構築と学習を分離する。

```python
import lightning as L
from torch.utils.data import DataLoader

from config.config import TrainConfig, app
from datasets.face_dataset import FaceDataset
from datasets.transforms import build_transform
from models.vit import ViTBackbone
from models.ir import IRBackbone
from models.classifier import PartialFC
from engine.trainer import FaceRecognitionModel


def build_model(cfg: TrainConfig):
    if cfg.model == "vit":
        return ViTBackbone(embedding_dim=cfg.embedding_dim)
    if cfg.model == "ir50":
        return IRBackbone(embedding_dim=cfg.embedding_dim)
    return IRBackbone(embedding_dim=cfg.embedding_dim)


def build_classifier(cfg: TrainConfig):
    return PartialFC(
        num_classes=cfg.num_classes,
        embedding_dim=cfg.embedding_dim,
        sample_rate=cfg.sample_rate,
    )


@app.default
def train(cfg: TrainConfig):
    transform = build_transform(image_size=cfg.image_size)
    dataset = FaceDataset(..., transform=transform)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = build_model(cfg)
    classifier = build_classifier(cfg)
    pl_model = FaceRecognitionModel(model=model, classifier=classifier, cfg=cfg)

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        default_root_dir=str(cfg.output_dir),
        logger=True,
        enable_checkpointing=True,
    )
    trainer.fit(pl_model, loader)


if __name__ == "__main__":
    app()
```

ポイント:

- 実行入口は `train.py`
- 設定は `config/config.py` に集約
- `build_model()` と `build_classifier()` で切り替えを管理する
- Trainer は学習ループだけに集中する

---

## 責務分離

### `datasets/face_dataset.py`

- 画像とラベルを返す
- `__getitem__` で画像と ID を組にして返す

```python
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self._load_image(self.image_paths[idx])
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
```

### `models/backbone.py`

- 画像から特徴を抽出する
- `embedding, feature_norm` を返す

### `models/classifier.py`

- Partial FC を抽象化する
- `embedding, labels` を受け取り logits を返す

### `losses/arcface.py` / `losses/adaface.py`

- ArcFace / AdaFace のロスを定義する
- Trainer 側では分岐を入れない

---

## 実装ルール

### ルール1: モデル生成ロジックは分離する

```python
def build_margin(cfg: TrainConfig):
    if cfg.margin == "arcface":
        return ArcFaceLoss(scale=cfg.scale, margin=cfg.margin_value)
    if cfg.margin == "adaface":
        return AdaFaceLoss(scale=cfg.scale, h=cfg.adaface_h)
    raise ValueError(f"Unsupported margin: {cfg.margin}")
```

### ルール2: Backbone は特徴抽出器に限定する

- 入力画像から特徴を生成する
- embedding と feature_norm を返す
- 分類ロジックや損失計算を持たない

### ルール3: Evaluator は学習ロジックと分離する

- 学習中に評価ロジックを混ぜない
- 学習後に `checkpoint` から読み込んで評価する

---

## 比較戦略

比較は次のように整理する。

```text
比較1: ViT-B + Full FC + ArcFace
比較2: ViT-B + Full FC + AdaFace
比較3: ViT-B + Partial FC + ArcFace
比較4: ViT-B + Partial FC + AdaFace
```

これにより、

- Backbone の効果
- Partial FC の効果
- ArcFace と AdaFace の差分
- 最終ベースライン選定

を明確に比較できる。

---

## まとめ

この骨組みの中心思想は次の通り。

- Backbone は特徴抽出に専念する
- Partial FC は ID を比較対象とする
- ArcFace / AdaFace は識別境界を決める
- Lightning が学習ループを管理する
- `config/config.py` が実験条件と CLI を一元管理する
- MLflow が実験の再現と比較を支える

この分離により、研究コードとして保守性、比較実験性、拡張性が高まる。