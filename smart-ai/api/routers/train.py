"""B5 — POST /api/v1/train（觸發訓練，BackgroundTasks 非同步執行）。"""
import codecs
import random
import sys
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel

from api.middleware.auth import verify_api_key

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "src"))

from db_connector import MySQLConnector  # noqa: E402

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(verify_api_key)])


class DBParams(BaseModel):
    host:     str
    port:     int = 3306
    user:     str
    password: str
    database: str
    table:    str


class TrainRequest(BaseModel):
    system: str
    org:    Optional[str] = None
    db:     DBParams


class TrainResponse(BaseModel):
    status:  str
    system:  str
    org:     Optional[str]
    version: str
    message: str


def _load_merged_yaml(systems_root: Path, system: str, org: Optional[str]) -> tuple[dict, Path]:
    """載入系統設定，org 設定覆蓋系統設定，回傳 (merged_dict, config_path)。"""
    sys_path = systems_root / system / "config.yaml"
    with open(sys_path, encoding="utf-8") as f:
        merged = yaml.safe_load(f)

    if org:
        org_path = systems_root / system / "orgs" / org / "config.yaml"
        with open(org_path, encoding="utf-8") as f:
            org_raw = yaml.safe_load(f)
        merged.update({k: v for k, v in org_raw.items() if v is not None})
        return merged, org_path
    return merged, sys_path


def _write_split(path: Path, rows: list[tuple[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with codecs.open(path, "w", encoding="utf-8") as f:
        for label, text in rows:
            f.write(f"{label}\t{text}\n")


def _run_training(system: str, org: Optional[str], db: DBParams, systems_root: Path):
    from config import Config
    from train import train as run_train

    version = datetime.now().strftime("v%Y%m%d")
    logger.info("[train] 開始：system=%s org=%s version=%s", system, org, version)

    # 1. 載入 config
    raw, config_path = _load_merged_yaml(systems_root, system, org)
    field_map    = raw.get("db", {}).get("field_map", {})
    input_fields = raw.get("input_fields", [])
    label_col    = field_map.get("label_unit")

    if not label_col:
        raise ValueError(f"config 缺少 db.field_map.label_unit：{config_path}")

    text_db_cols = [field_map[f] for f in input_fields if f in field_map]
    if not text_db_cols:
        raise ValueError(f"config input_fields 無對應 field_map 欄位：{config_path}")

    # 2. 連接 DB，抓取資料
    connector = MySQLConnector(
        host=db.host, port=db.port,
        user=db.user, password=db.password,
        database=db.database,
    )
    connector.connect()
    connector.execute("SELECT 1")
    logger.info("[train] DB 連線驗證成功")

    cols_sql = ", ".join(text_db_cols + [label_col])
    rows = connector.execute(f"SELECT {cols_sql} FROM {db.table}")
    connector.close()

    if not rows:
        raise ValueError(f"DB 查無資料：{db.table}")
    logger.info("[train] 取得 %d 筆資料", len(rows))

    # 3. 切分並寫入資料檔
    n_text = len(text_db_cols)
    data = [(" ".join(str(r[i]) for i in range(n_text)), str(r[n_text])) for r in rows]
    # data: [(text, label), ...]
    random.shuffle(data)
    n = len(data)
    splits = {
        "train": [(lbl, txt) for txt, lbl in data[:int(0.8 * n)]],
        "val":   [(lbl, txt) for txt, lbl in data[int(0.8 * n):int(0.9 * n)]],
        "test":  [(lbl, txt) for txt, lbl in data[int(0.9 * n):]],
    }

    paths_raw = raw.get("paths", {})
    data_dir  = Path(paths_raw.get("data_dir", "/app/data"))
    _write_split(data_dir / paths_raw.get("train_file", "train.txt"), splits["train"])
    _write_split(data_dir / paths_raw.get("val_file",   "val.txt"),   splits["val"])
    _write_split(data_dir / paths_raw.get("test_file",  "test.txt"),  splits["test"])
    logger.info("[train] 資料檔寫入完成：%s", data_dir)

    # 4. 訓練 — 輸出至版本目錄
    cfg = Config(str(config_path))
    versioned_dir = cfg.model_dir / version
    versioned_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_dir   = versioned_dir
    cfg.labels_path = versioned_dir / cfg.labels_path.name

    run_train(cfg)
    logger.info("[train] 訓練完成：%s", versioned_dir)

    # 5. 更新 current 軟連結
    current_link = cfg.model_dir.parent / "current"
    if current_link.is_symlink():
        current_link.unlink()
    current_link.symlink_to(version, target_is_directory=True)
    logger.info("[train] current 軟連結更新：current -> %s", version)


@router.post("/train", status_code=202, response_model=TrainResponse)
def trigger_train(req: TrainRequest, background_tasks: BackgroundTasks, request: Request):
    systems_root: Path = request.app.state.systems_root
    version = datetime.now().strftime("v%Y%m%d")

    background_tasks.add_task(
        _run_training, req.system, req.org, req.db, systems_root
    )
    return TrainResponse(
        status="accepted",
        system=req.system,
        org=req.org,
        version=version,
        message=f"訓練任務已排入，完成後模型將更新至 checkpoints/bert/{version}/",
    )
