"""
Phase A.2 smoke test — 驗證 config.yaml 可被 Config() 正確載入。

【驗收條件】
- 載入 Config() 不丟例外
- 所有欄位值等於 yaml 原始內容
- Path 欄位型別為 pathlib.Path，且字串符合預期組合

【執行方式（L1，不需 GPU/torch）】
  cd smart-ai
  python -m pytest tests/test_config_load.py -v
  # 或不裝 pytest：
  python tests/test_config_load.py

【關於原 status.md 驗收條件「dry-run 一個 epoch」】
那是 L2 端到端驗證，需要 GPU 機器 + 訓練資料；L1 通過已能證明
「設定外推不丟失資訊」這個 Phase A.2 的核心目標。
"""
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import Config, DEFAULT_CONFIG_PATH  # noqa: E402


def _load_yaml() -> dict:
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_config_loads_without_error():
    cfg = Config()
    assert cfg is not None


def test_model_fields_match_yaml():
    y = _load_yaml()
    cfg = Config()
    assert cfg.pretrained_model == y["model"]["pretrained"]
    assert cfg.num_classes == y["model"]["num_classes"]
    assert cfg.max_length == y["model"]["max_length"]


def test_training_fields_match_yaml():
    y = _load_yaml()
    cfg = Config()
    t = y["training"]
    assert cfg.epochs == t["epochs"]
    assert cfg.batch_size == t["batch_size"]
    assert cfg.learning_rate == float(t["learning_rate"])
    assert cfg.warmup_ratio == t["warmup_ratio"]
    assert cfg.weight_decay == t["weight_decay"]


def test_path_fields_are_path_and_composed_correctly():
    y = _load_yaml()
    p = y["paths"]
    cfg = Config()

    for attr in ("train_file", "val_file", "test_file", "model_dir", "labels_path"):
        assert isinstance(getattr(cfg, attr), Path), f"{attr} 應為 Path 型別"

    assert cfg.train_file == Path(p["data_dir"]) / p["train_file"]
    assert cfg.val_file == Path(p["data_dir"]) / p["val_file"]
    assert cfg.test_file == Path(p["data_dir"]) / p["test_file"]
    assert cfg.model_dir == Path(p["checkpoint_dir"]) / p["model_subdir"]
    assert cfg.labels_path == Path(p["checkpoint_dir"]) / p["labels_file"]


def test_learning_rate_is_float_not_string():
    cfg = Config()
    assert isinstance(cfg.learning_rate, float)
    assert cfg.learning_rate > 0


def _run_all():
    tests = [
        test_config_loads_without_error,
        test_model_fields_match_yaml,
        test_training_fields_match_yaml,
        test_path_fields_are_path_and_composed_correctly,
        test_learning_rate_is_float_not_string,
    ]
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print()
    if failed:
        print(f"[smoke] {failed} 個測試失敗")
        sys.exit(1)
    print(f"[smoke] 全部 {len(tests)} 個測試通過 — Phase A.2 L1 驗收 OK")


if __name__ == "__main__":
    print(f"[smoke] yaml 路徑：{DEFAULT_CONFIG_PATH}")
    print(f"[smoke] 開始驗證 Config() 載入...\n")
    _run_all()
