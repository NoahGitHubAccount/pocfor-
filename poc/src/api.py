"""
FastAPI 服務入口。

提供兩個端點（對應舊系統的 Flask 路由）：
  POST /predict  → 文本分類，回傳 Top-N 局處
  POST /tfidf    → jieba TF-IDF 關鍵詞萃取

模型在啟動時載入一次（lifespan），之後每次 predict 都很快。
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import jieba.analyse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from predict import Predictor


# ─── 應用程式生命週期：啟動時載入模型 ────────────────────────

_predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan：啟動時初始化，關閉時清理。"""
    global _predictor
    cfg = Config()
    if cfg.model_path.exists():
        _predictor = Predictor(cfg)
        print("[api] 模型載入完成，服務就緒")
    else:
        print(f"[api] 警告：找不到模型檔案 {cfg.model_path}，/predict 將回傳錯誤")
    yield
    _predictor = None


app = FastAPI(
    title="智慧分案 API",
    version="1.0.0",
    description="輸入陳情文字，自動預測應分配的局處（TextCNN）",
    lifespan=lifespan,
)


# ─── Request / Response Schema ────────────────────────────────

class PredictRequest(BaseModel):
    preString: str  = Field(..., description="陳情文字")
    preNum:    int  = Field(3,   description="回傳幾個候選結果", ge=1, le=9)


class TFIDFRequest(BaseModel):
    preString: str = Field(..., description="文字內容")
    preNum:    int = Field(5,   description="回傳幾個關鍵詞",   ge=1, le=50)


# ─── 端點 ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _predictor is not None,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """
    預測文本分類，回傳 Top-N 局處與信心值。

    範例請求：
        {"preString": "路燈故障無法修復", "preNum": 3}
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="模型尚未載入，請先執行訓練")

    results = _predictor.predict(req.preString, top_n=req.preNum)
    return {"case": results}


@app.post("/tfidf")
def tfidf(req: TFIDFRequest):
    """
    jieba TF-IDF 關鍵詞萃取。

    範例請求：
        {"preString": "路燈故障，建請環保局處理", "preNum": 3}
    """
    keywords = jieba.analyse.extract_tags(
        req.preString,
        topK=req.preNum,
        withWeight=True,
        allowPOS=(),
    )
    results = [
        {"ou": word, "probability": f"{weight:.10f}"}
        for word, weight in keywords
    ]
    return {"case": results}
