"""
Phase B L1 smoke test — 不需 GPU / 模型權重。

【執行方式】
  cd smart-ai
  python -m pytest tests/test_api.py -v
  # 或：
  python tests/test_api.py
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("SMART_AI_API_KEY", "test-key")
os.environ.setdefault("SMART_AI_SYSTEMS_ROOT", str(ROOT / "systems"))

from fastapi.testclient import TestClient  # noqa: E402
from api.main import create_app            # noqa: E402

HEADERS = {"X-API-Key": "test-key"}


def _make_client():
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


# ── /health ────────────────────────────────────────────────────────────────

def test_health():
    with _make_client() as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ── auth ───────────────────────────────────────────────────────────────────

def test_predict_no_key_returns_401():
    with _make_client() as client:
        r = client.post("/api/v1/predict",
                        json={"system": "taitung_bigdata", "data": {"text": "test"}})
        assert r.status_code == 401


def test_predict_wrong_key_returns_401():
    with _make_client() as client:
        r = client.post("/api/v1/predict",
                        json={"system": "taitung_bigdata", "data": {"text": "test"}},
                        headers={"X-API-Key": "wrong"})
        assert r.status_code == 401


# ── request validation ─────────────────────────────────────────────────────

def test_predict_missing_system_returns_422():
    with _make_client() as client:
        r = client.post("/api/v1/predict",
                        json={"data": {"text": "test"}},
                        headers=HEADERS)
        assert r.status_code == 422


def test_batch_missing_data_returns_422():
    with _make_client() as client:
        r = client.post("/api/v1/batch",
                        json={"system": "taitung_bigdata"},
                        headers=HEADERS)
        assert r.status_code == 422


def test_train_missing_db_returns_422():
    with _make_client() as client:
        r = client.post("/api/v1/train",
                        json={"system": "taitung_bigdata"},
                        headers=HEADERS)
        assert r.status_code == 422


# ── predict with mocked model ──────────────────────────────────────────────

def test_predict_with_mock_model():
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = [
        {"ou": "環保局", "probability": "0.9500000000"},
        {"ou": "建設局", "probability": "0.0500000000"},
    ]
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.model_manager = MagicMock()
        app.state.model_manager.get.return_value = mock_predictor

        r = client.post("/api/v1/predict",
                        json={"system": "taitung_bigdata",
                              "data": {"text": "道路破損請修繕"}},
                        headers=HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert body["system"] == "taitung_bigdata"
        assert len(body["predictions"]) == 2
        assert body["predictions"][0]["label"] == "環保局"


# ── train returns 202 ──────────────────────────────────────────────────────

def test_train_accepted():
    with _make_client() as client:
        r = client.post("/api/v1/train",
                        json={
                            "system": "chiefmail_back",
                            "org": "hpa",
                            "db": {
                                "host": "127.0.0.1", "port": 3306,
                                "user": "u", "password": "p",
                                "database": "d", "table": "t",
                            },
                        },
                        headers=HEADERS)
        assert r.status_code == 202
        assert r.json()["status"] == "accepted"


# ── runner ─────────────────────────────────────────────────────────────────

def _run_all():
    tests = [
        test_health,
        test_predict_no_key_returns_401,
        test_predict_wrong_key_returns_401,
        test_predict_missing_system_returns_422,
        test_batch_missing_data_returns_422,
        test_train_missing_db_returns_422,
        test_predict_with_mock_model,
        test_train_accepted,
    ]
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print()
    if failed:
        print(f"[smoke] {failed} 個測試失敗")
        sys.exit(1)
    print(f"[smoke] 全部 {len(tests)} 個測試通過 — Phase B L1 驗收 OK")


if __name__ == "__main__":
    print("[smoke] Phase B API L1 驗收...\n")
    _run_all()
