"""
模型與訓練的所有參數集中管理。
修改這裡就等於修改整個系統的行為。
"""
from dataclasses import dataclass, field
from pathlib import Path

# 專案根目錄（/app 為容器內路徑）
BASE_DIR = Path("/app")
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"


@dataclass
class Config:
    # ── 資料路徑 ──────────────────────────────────────────
    train_file: Path = DATA_DIR / "cnews.train.txt"
    val_file:   Path = DATA_DIR / "cnews.val.txt"
    test_file:  Path = DATA_DIR / "cnews.test.txt"
    vocab_file: Path = DATA_DIR / "vocab.txt"
    vector_npz: Path = DATA_DIR / "vector_word.npz"

    # ── 模型產出路徑 ──────────────────────────────────────
    model_path: Path = CHECKPOINT_DIR / "best_model.pt"
    labels_path: Path = CHECKPOINT_DIR / "labels.txt"

    # ── 模型架構 ──────────────────────────────────────────
    embedding_dim: int = 100       # 詞向量維度（對應舊系統 embedding_size）
    vocab_size: int = 8000         # 詞彙表大小
    num_classes: int = 9           # 分類數（9 個局處）
    seq_length: int = 600          # 句子最大詞數
    num_filters: int = 128         # 每種卷積核的數量
    filter_sizes: list = field(default_factory=lambda: [2, 3, 4])  # 卷積核大小

    # ── 訓練參數 ──────────────────────────────────────────
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    dropout: float = 0.5
    early_stop_patience: int = 5   # 驗證集多少 epoch 沒進步就停止
