# 智慧分案 POC

市民陳情文字自動分類系統的概念驗證（Proof of Concept）。輸入一段陳情文字，模型預測應分派的局處，提供 Top-N 候選結果與信心值。

## 專案架構

```
├── poc/                # 方向 A — PyTorch TextCNN
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── src/
│       ├── api.py            # FastAPI 服務（POST /predict, /tfidf）
│       ├── train.py          # 訓練腳本
│       ├── predict.py        # 推論邏輯
│       ├── model.py          # TextCNN 模型定義
│       ├── data_loader.py    # 資料前處理 & DataLoader
│       ├── eval.py           # 評估腳本
│       ├── word2vec_helper.py
│       └── config.py         # 超參數集中管理
│
├── poc-bert/           # 方向 B — BERT 中文微調
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── src/
│       ├── api.py            # FastAPI 服務（同介面，Port 8081）
│       ├── train.py
│       ├── predict.py
│       ├── model.py          # HuggingFace BERT 微調
│       ├── data_loader.py
│       ├── eval.py
│       └── config.py
│
├── docs/               # 技術文件與操作手冊
├── notes/              # 開發筆記
└── PLAN.md             # 專案計畫與進度追蹤
```

## 兩種方案比較

| 項目 | TextCNN (`poc/`) | BERT (`poc-bert/`) |
|------|------------------|--------------------|
| 預訓練模型 | Word2Vec (自訓) | `hfl/chinese-roberta-wwm-ext` |
| 斷詞 | jieba 手動斷詞 | BERT Tokenizer（自動） |
| 訓練 Epochs | 30 | 3–5 |
| 模型大小 | ~4 MB | ~400 MB |
| 推論速度 | 極快 | < 1 秒/筆 |
| API Port | 8080 | 8081 |

兩個方案可同時運行，互不衝突。

## 快速開始

### 環境需求

- Docker & Docker Compose
- GPU 機器（訓練用；推論可 CPU-only）

### 訓練

```bash
# TextCNN
cd poc
docker compose run --rm train

# BERT
cd poc-bert
docker compose run --rm train
```

訓練完成後，模型 checkpoint 會產生在各自的 `checkpoints/` 目錄。

### 啟動 API 服務

```bash
# TextCNN（Port 8080）
cd poc
docker compose up -d api

# BERT（Port 8081）
cd poc-bert
docker compose up -d api
```

### API 使用範例

```bash
# 預測局處
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"preString": "路燈故障無法修復", "preNum": 3}'

# TF-IDF 關鍵詞萃取（僅 TextCNN 版）
curl -X POST http://localhost:8080/tfidf \
  -H "Content-Type: application/json" \
  -d '{"preString": "路燈故障，建請環保局處理", "preNum": 3}'

# 健康檢查
curl http://localhost:8080/health
```

## 技術文件

- [自動化訓練文件](docs/01_自動化訓練文件.md)
- [預測 API 文件](docs/02_預測API文件.md)
- [分案系統 README](docs/03_分案系統README.md)
- [GPU 訓練操作手冊（TextCNN）](docs/GPU訓練操作手冊.md)
- [GPU 訓練操作手冊（BERT）](docs/GPU訓練操作手冊_BERT.md)
