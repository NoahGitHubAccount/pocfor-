"""
Phase A6 smoke test — db_connector.py L1 驗收（不需真實 DB）

【執行方式】
  cd smart-ai
  python tests/test_db_connector.py
  # 或：
  python -m pytest tests/test_db_connector.py -v

【L2 端對端測試（需真實 MySQL）】
  設定環境變數後執行 test_mysql_select_1()
  DB_HOST=... DB_USER=... DB_PASSWORD=... DB_NAME=... python tests/test_db_connector.py --l2
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

from db_connector import DBConnector, MySQLConnector, MSSQLConnector  # noqa: E402


def test_dbconnector_is_abstract():
    import inspect
    assert inspect.isabstract(DBConnector)


def test_mysqlconnector_is_concrete():
    import inspect
    assert not inspect.isabstract(MySQLConnector)


def test_mssql_connect_raises():
    conn = MSSQLConnector()
    try:
        conn.connect()
        assert False, "應丟出 NotImplementedError"
    except NotImplementedError:
        pass


def test_mssql_execute_raises():
    conn = MSSQLConnector()
    try:
        conn.execute("SELECT 1")
        assert False, "應丟出 NotImplementedError"
    except NotImplementedError:
        pass


def _run_all():
    tests = [
        test_dbconnector_is_abstract,
        test_mysqlconnector_is_concrete,
        test_mssql_connect_raises,
        test_mssql_execute_raises,
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
    print(f"[smoke] 全部 {len(tests)} 個測試通過 — Phase A6 L1 驗收 OK")


if __name__ == "__main__":
    print("[smoke] Phase A6 db_connector L1 驗收...\n")
    _run_all()
