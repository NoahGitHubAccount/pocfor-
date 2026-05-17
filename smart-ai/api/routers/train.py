"""B5 — POST /api/v1/train（觸發訓練，BackgroundTasks 非同步執行）。"""
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel

from api.middleware.auth import verify_api_key

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "core"))

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


def _run_training(system: str, org: Optional[str], db: DBParams, systems_root: Path):
    version = datetime.now().strftime("v%Y%m%d")
    logger.info("[train] 開始：system=%s org=%s version=%s", system, org, version)

    # 1. 驗證 DB 連線
    connector = MySQLConnector(
        host=db.host, port=db.port,
        user=db.user, password=db.password,
        database=db.database,
    )
    connector.connect()
    connector.execute("SELECT 1")
    connector.close()
    logger.info("[train] DB 連線驗證成功")

    # 2. TODO：從 DB 抓取訓練資料並寫入暫存檔
    # 3. TODO：呼叫 src/train.py 執行訓練
    # 4. TODO：將 checkpoint 寫入 checkpoints/bert/{version}/
    # 5. TODO：更新 current 軟連結

    logger.info("[train] 完成（stub）：version=%s", version)


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
