FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch（避免把 CUDA 版拉進來；GPU 機器訓練時改 cu118/cu121）
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/     ./src/
COPY core/    ./core/
COPY api/     ./api/
COPY systems/ ./systems/

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
