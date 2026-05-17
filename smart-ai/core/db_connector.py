"""
DB 連線統一介面。

連線參數從環境變數讀取：
  DB_HOST, DB_PORT（預設 3306）, DB_USER, DB_PASSWORD, DB_NAME

使用方式：
  conn = MySQLConnector()
  conn.connect()
  rows = conn.execute("SELECT 1")
  conn.close()
"""
import os
from abc import ABC, abstractmethod
from typing import Any, Optional


class DBConnector(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def execute(self, sql: str, params: Optional[tuple] = None) -> Any: ...

    @abstractmethod
    def close(self) -> None: ...


class MySQLConnector(DBConnector):
    def __init__(self):
        self.host     = os.environ["DB_HOST"]
        self.port     = int(os.environ.get("DB_PORT", 3306))
        self.user     = os.environ["DB_USER"]
        self.password = os.environ["DB_PASSWORD"]
        self.database = os.environ["DB_NAME"]
        self._conn    = None

    def connect(self) -> None:
        import pymysql
        self._conn = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset="utf8mb4",
        )

    def execute(self, sql: str, params: Optional[tuple] = None) -> Any:
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


class MSSQLConnector(DBConnector):
    """MSSQL 預留介面，尚未實作。等 chiefmail_back 接入需求後補齊。"""

    def connect(self) -> None:
        raise NotImplementedError("MSSQLConnector 尚未實作")

    def execute(self, sql: str, params: Optional[tuple] = None) -> Any:
        raise NotImplementedError("MSSQLConnector 尚未實作")

    def close(self) -> None:
        raise NotImplementedError("MSSQLConnector 尚未實作")
