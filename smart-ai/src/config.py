"""
BERT 方案的參數設定（從 config.yaml 載入）。

向後兼容：對外仍提供 `Config` class 與相同欄位名稱，
callers（train / predict / api / data_loader / eval / model）不需修改 import。

yaml 路徑解析優先序：
1. 環境變數 `SMART_AI_CONFIG`（絕對路徑）
2. 預設：`smart-ai/systems/taitung_bigdata/config.yaml`
"""
from pathlib import Path
import os
import yaml

BASE_DIR = Path("/app")

DEFAULT_CONFIG_PATH = os.environ.get(
    "SMART_AI_CONFIG",
    str(Path(__file__).resolve().parent.parent / "systems" / "taitung_bigdata" / "config.yaml"),
)


class Config:
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        with open(config_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)

        paths = y["paths"]
        data_dir = Path(paths["data_dir"])
        ckpt_dir = Path(paths["checkpoint_dir"])

        # ── 資料路徑 ──────────────────────────────────────────
        self.train_file = data_dir / paths["train_file"]
        self.val_file   = data_dir / paths["val_file"]
        self.test_file  = data_dir / paths["test_file"]

        # ── 模型產出路徑 ──────────────────────────────────────
        self.model_dir   = ckpt_dir / paths["model_subdir"]
        self.labels_path = ckpt_dir / paths["labels_file"]

        # ── 預訓練模型 ────────────────────────────────────────
        self.pretrained_model = y["model"]["pretrained"]

        # ── 模型參數 ──────────────────────────────────────────
        self.num_classes = y["model"]["num_classes"]
        self.max_length  = y["model"]["max_length"]

        # ── 訓練參數 ──────────────────────────────────────────
        t = y["training"]
        self.epochs        = t["epochs"]
        self.batch_size    = t["batch_size"]
        self.learning_rate = float(t["learning_rate"])
        self.warmup_ratio  = t["warmup_ratio"]
        self.weight_decay  = t["weight_decay"]
