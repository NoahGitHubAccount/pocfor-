"""
BERT 版 FastAPI 服務。
API 介面和 TextCNN 版完全相同（/predict、/tfidf），方便前端無縫切換。
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

_predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    cfg = Config()
    if cfg.model_dir.exists() and (cfg.model_dir / "config.json").exists():
        _predictor = Predictor(cfg)
        print("[api] BERT 模型載入完成，服務就緒")
    else:
        print(f"[api] 警告：找不到 BERT 模型 {cfg.model_dir}，/predict 將回傳錯誤")
    yield
    _predictor = None


app = FastAPI(
    title="智慧分案 API（BERT 版）",
    version="2.0.0",
    description="輸入陳情文字，自動預測應分配的局處（BERT fine-tuned）",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    preString: str = Field(..., description="陳情文字")
    preNum:    int = Field(3, description="回傳幾個候選結果", ge=1, le=9)


class TFIDFRequest(BaseModel):
    preString: str = Field(..., description="文字內容")
    preNum:    int = Field(5, description="回傳幾個關鍵詞", ge=1, le=50)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _predictor is not None,
        "model_type": "BERT (chinese-roberta-wwm-ext)",
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="BERT 模型尚未載入，請先執行訓練")
    results = _predictor.predict(req.preString, top_n=req.preNum)
    return {"case": results}


@app.post("/tfidf")
def tfidf(req: TFIDFRequest):
    keywords = jieba.analyse.extract_tags(
        req.preString, topK=req.preNum, withWeight=True, allowPOS=()
    )
    return {
        "case": [
            {"ou": word, "probability": f"{weight:.10f}"}
            for word, weight in keywords
        ]
    }
