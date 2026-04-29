FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先裝 PyTorch CPU，避免後續 transformers 安裝時把 CUDA 版拉進來
# GPU 機器訓練時改為 cu118/cu121
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu

# 其他套件（transformers 會沿用已裝好的 torch，不會再下載）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8081"]
