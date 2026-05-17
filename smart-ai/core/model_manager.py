"""
B6 — 模型載入與快取管理。

每個 (system, org) 組合對應一個 Predictor 實例，首次呼叫時 lazy load，
之後直接從快取回傳。prod 環境只會有一個 system，快取永遠只有一筆。
"""
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class ModelManager:
    def __init__(self, systems_root: Path):
        self._systems_root = systems_root
        self._cache: dict = {}

    def get(self, system: str, org: Optional[str] = None):
        # lazy import：torch/transformers 只在真正載入模型時才引入
        from config import Config
        from predict import Predictor

        key = f"{system}/{org}" if org else system
        if key not in self._cache:
            config_path = self._config_path(system, org)
            if not config_path.exists():
                raise FileNotFoundError(f"找不到 config：{config_path}")
            self._cache[key] = Predictor(Config(str(config_path)))
        return self._cache[key]

    def _config_path(self, system: str, org: Optional[str]) -> Path:
        if org:
            return self._systems_root / system / "orgs" / org / "config.yaml"
        return self._systems_root / system / "config.yaml"
