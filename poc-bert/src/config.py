"""
BERT 方案的參數設定。

【白話說明 — BERT vs TextCNN 差異】
TextCNN 需要自己訓練詞向量、自己管理詞彙表。
BERT 已經在海量中文語料上預訓練好了，內建 tokenizer，
我們只需要做「微調」（fine-tune）：
  用我們的分案資料，讓 BERT 學會「哪類文字 → 哪個局處」。

選用 `hfl/chinese-roberta-wwm-ext` 而非 `bert-base-chinese` 的原因：
  它用全詞遮蔽（Whole Word Masking）訓練，
  對中文斷詞更友善，下游任務效果更好。
"""
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path("/app")
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"


@dataclass
class Config:
    # ── 資料路徑 ──────────────────────────────────────────
    train_file: Path = DATA_DIR / "cnews.train.txt"
    val_file:   Path = DATA_DIR / "cnews.val.txt"
    test_file:  Path = DATA_DIR / "cnews.test.txt"

    # ── 模型產出路徑 ──────────────────────────────────────
    model_dir:   Path = CHECKPOINT_DIR / "bert-model"
    labels_path: Path = CHECKPOINT_DIR / "labels.txt"

    # ── 預訓練模型名稱（HuggingFace Hub）─────────────────
    pretrained_model: str = "hfl/chinese-roberta-wwm-ext"

    # ── 模型參數 ──────────────────────────────────────────
    num_classes:  int = 9
    max_length:   int = 512     # BERT 最大 token 長度（上限 512）

    # ── 訓練參數 ──────────────────────────────────────────
    epochs:        int   = 5    # BERT 通常 3-5 epochs 就收斂
    batch_size:    int   = 16   # BERT 比較大，batch 要小（GPU 記憶體限制）
    learning_rate: float = 2e-5 # BERT fine-tune 典型學習率
    warmup_ratio:  float = 0.1  # 前 10% steps 線性 warmup
    weight_decay:  float = 0.01
