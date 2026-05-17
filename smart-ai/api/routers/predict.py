"""B3 — POST /api/v1/predict（即時推論，單筆）。"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from api.middleware.auth import verify_api_key

router = APIRouter(dependencies=[Depends(verify_api_key)])


class PredictRequest(BaseModel):
    system: str
    org:    Optional[str] = None
    data:   dict[str, str]
    top_n:  int = 3


class Prediction(BaseModel):
    label:      str
    confidence: float


class PredictResponse(BaseModel):
    system:      str
    org:         Optional[str]
    predictions: list[Prediction]


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    manager = request.app.state.model_manager
    try:
        predictor = manager.get(req.system, req.org)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"模型載入失敗：{e}")

    text = " ".join(str(v) for v in req.data.values())
    raw = predictor.predict(text, top_n=req.top_n)
    return PredictResponse(
        system=req.system,
        org=req.org,
        predictions=[
            Prediction(label=r["ou"], confidence=float(r["probability"]))
            for r in raw
        ],
    )
