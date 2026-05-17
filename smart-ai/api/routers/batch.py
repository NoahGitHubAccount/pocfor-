"""B4 — POST /api/v1/batch（批次推論）。"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from api.middleware.auth import verify_api_key

router = APIRouter(dependencies=[Depends(verify_api_key)])


class BatchRequest(BaseModel):
    system: str
    org:    Optional[str] = None
    data:   list[dict[str, str]]
    top_n:  int = 3


class Prediction(BaseModel):
    label:      str
    confidence: float


class BatchResponse(BaseModel):
    system:  str
    org:     Optional[str]
    results: list[list[Prediction]]


@router.post("/batch", response_model=BatchResponse)
def batch(req: BatchRequest, request: Request):
    manager = request.app.state.model_manager
    try:
        predictor = manager.get(req.system, req.org)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"模型載入失敗：{e}")

    results = []
    for item in req.data:
        text = " ".join(str(v) for v in item.values())
        raw = predictor.predict(text, top_n=req.top_n)
        results.append([
            Prediction(label=r["ou"], confidence=float(r["probability"]))
            for r in raw
        ])
    return BatchResponse(system=req.system, org=req.org, results=results)
