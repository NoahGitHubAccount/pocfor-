FROM python:3.11-slim

WORKDIR /app

# 系統依賴（gensim/numpy 編譯需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先裝非 PyTorch 套件（從 PyPI，快）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再裝 PyTorch CPU-only（獨立 index，避免互相干擾）
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu

# 複製程式碼
COPY src/ ./src/

# 預設啟動 API（train service 會透過 docker-compose 覆寫）
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
