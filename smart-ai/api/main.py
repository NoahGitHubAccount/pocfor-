"""B1 — FastAPI 主程式，多系統路由入口。"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

from model_manager import ModelManager  # noqa: E402

SYSTEMS_ROOT = Path(os.environ.get(
    "SMART_AI_SYSTEMS_ROOT",
    str(ROOT / "systems"),
))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_manager = ModelManager(SYSTEMS_ROOT)
    app.state.systems_root  = SYSTEMS_ROOT
    yield
    app.state.model_manager = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="smart-ai API",
        version="1.0.0",
        description="多系統、多機關 AI 推論與訓練服務",
        lifespan=lifespan,
    )

    from api.routers import predict, batch, train
    app.include_router(predict.router, prefix="/api/v1")
    app.include_router(batch.router,   prefix="/api/v1")
    app.include_router(train.router,   prefix="/api/v1")

    @app.get("/health")
    def health():
        return {"status": "ok", "systems_root": str(SYSTEMS_ROOT)}

    return app


app = create_app()
