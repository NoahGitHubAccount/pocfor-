"""B2 — API Key 驗證（內網後台對後台，單一 key 即可）。"""
import os
from typing import Optional
from fastapi import Header, HTTPException


async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    expected = os.environ.get("SMART_AI_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=500, detail="伺服器未設定 SMART_AI_API_KEY")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API Key")
