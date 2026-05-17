"""
C1 — 定期訓練排程。

排程設定從 scheduler.yaml 讀取，格式：
  jobs:
    - system: taitung_bigdata
      org: ~
      cron: "0 2 * * 0"   # 分 時 日 月 週（每週日 02:00）
      db:
        host: 127.0.0.1
        port: 3306
        user: smart_ai
        database: smart_ai_db
        table: case_records

DB 密碼從環境變數讀取：
  優先：DB_PASSWORD_{SYSTEM}（大寫）
  fallback：DB_PASSWORD
"""
import os
import sys
import logging
import yaml
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


class TrainingScheduler:
    def __init__(self, systems_root: Path, config_path: Path):
        self._systems_root = systems_root
        self._config_path  = config_path
        self._scheduler    = BackgroundScheduler(timezone="Asia/Taipei")

    def start(self):
        self._load_jobs()
        self._scheduler.start()
        logger.info("[scheduler] 啟動，已載入 %d 個訓練排程",
                    len(self._scheduler.get_jobs()))

    def shutdown(self):
        self._scheduler.shutdown(wait=False)
        logger.info("[scheduler] 已關閉")

    def get_jobs(self) -> list[dict]:
        return [
            {"id": job.id, "next_run": str(job.next_run_time)}
            for job in self._scheduler.get_jobs()
        ]

    def _load_jobs(self):
        with open(self._config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        for job_cfg in cfg.get("jobs", []):
            self._register_job(job_cfg)

    def _register_job(self, job_cfg: dict):
        from api.routers.train import _run_training, DBParams

        system: str          = job_cfg["system"]
        org: Optional[str]   = job_cfg.get("org")
        cron: str            = job_cfg["cron"]
        db_raw: dict         = job_cfg["db"]

        password = os.environ.get(
            f"DB_PASSWORD_{system.upper()}",
            os.environ.get("DB_PASSWORD", ""),
        )

        db_params = DBParams(
            host     = db_raw["host"],
            port     = int(db_raw.get("port", 3306)),
            user     = db_raw["user"],
            password = password,
            database = db_raw["database"],
            table    = db_raw["table"],
        )

        minute, hour, day, month, dow = cron.strip().split()
        job_id = f"{system}__{org or 'default'}"

        self._scheduler.add_job(
            _run_training,
            trigger    = "cron",
            args       = [system, org, db_params, self._systems_root],
            id         = job_id,
            minute     = minute,
            hour       = hour,
            day        = day,
            month      = month,
            day_of_week= dow,
            replace_existing = True,
        )
        logger.info("[scheduler] 已登錄：%s  cron=%s", job_id, cron)
