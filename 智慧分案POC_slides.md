---
title: 智慧分案 POC
subtitle: 市民陳情文字自動分類系統概念驗證
author: 資訊服務團隊
date: 2026-04-27
---

# 專案概述

## 系統簡介

- 目標：市民陳情文字自動分類至對應局處
- 輸入：一段陳情文字
- 輸出：Top-N 候選局處與信心值
- 階段：概念驗證（Proof of Concept）

::: notes
本系統旨在透過自然語言處理技術，將市民陳情文字自動分類至對應的政府局處，提供 Top-N 候選結果與信心值，協助分案人員快速決策。
:::

# 專案架構

## 兩大技術方向

- **方向 A**：`poc/` — PyTorch TextCNN
- **方向 B**：`poc-bert/` — BERT 中文微調
- 兩方案可同時運行，互不衝突
- 統一 API 介面，便於比較與整合

::: notes
專案設計兩種並行技術路線：輕量級的 TextCNN 方案與重量級的 BERT 方案，讓使用單位可依實際需求選擇合適的模型。
:::

## 方向 A：TextCNN {layout="Two Content"}

**核心元件**

- FastAPI 服務
- POST /predict
- POST /tfidf
- 訓練 / 推論 / 評估腳本

**技術特色**

- PyTorch TextCNN 模型
- jieba 手動斷詞
- Word2Vec 自訓詞向量
- 模型大小約 4 MB

::: notes
TextCNN 方案使用 PyTorch 實作，搭配 jieba 斷詞與自訓練 Word2Vec 詞向量。優點是模型輕量（約 4 MB）、推論極快，適合資源受限的部署環境。
:::

## 方向 B：BERT 中文微調 {layout="Two Content"}

**核心元件**

- FastAPI 服務（Port 8081）
- POST /predict
- 訓練 / 推論 / 評估腳本
- HuggingFace BERT 微調

**技術特色**

- chinese-roberta-wwm-ext
- BERT Tokenizer 自動斷詞
- 訓練僅需 3–5 個 Epochs
- 推論 < 1 秒/筆

::: notes
BERT 方案採用 hfl/chinese-roberta-wwm-ext 預訓練模型進行微調，無需手動斷詞，準確率通常優於 TextCNN。模型大小約 400 MB，推論速度仍可達 < 1 秒/筆。
:::

# 兩種方案比較

## 方案規格對照

| 項目 | TextCNN | BERT |
|------|---------|------|
| 預訓練模型 | Word2Vec（自訓）| chinese-roberta-wwm-ext |
| 斷詞方式 | jieba 手動 | BERT Tokenizer（自動）|
| 訓練 Epochs | 30 | 3–5 |
| 模型大小 | ~4 MB | ~400 MB |
| 推論速度 | 極快 | < 1 秒/筆 |
| API Port | 8080 | 8081 |

::: notes
兩方案各有優勢：TextCNN 輕量快速，適合邊緣部署；BERT 準確率高，適合追求精準度的場景。兩者 API 介面相同，可無縫切換。
:::

# 快速開始

## 環境需求與訓練 {layout="Two Content"}

**環境需求**

- Docker & Docker Compose
- GPU 機器（訓練時需要）
- CPU-only（推論可用）

**訓練指令**

- TextCNN：
  `cd poc`
  `docker compose run --rm train`
- BERT：
  `cd poc-bert`
  `docker compose run --rm train`

::: notes
訓練完成後，模型 checkpoint 會產生在各自的 checkpoints/ 目錄。GPU 機器僅訓練時需要，推論階段可使用 CPU-only 環境。
:::

## 啟動 API 服務 {layout="Two Content"}

**TextCNN（Port 8080）**

```
cd poc
docker compose up -d api
```

- 提供 /predict、/tfidf
- 輕量快速部署

**BERT（Port 8081）**

```
cd poc-bert
docker compose up -d api
```

- 提供 /predict
- 高精準度推論

::: notes
兩個 API 服務使用相同的 RESTful 介面設計，僅端口不同（8080 vs 8081），可同時運行進行比較測試。
:::

## API 功能說明

- **POST /predict** — 輸入陳情文字，回傳 Top-N 局處與信心值
- **POST /tfidf** — TF-IDF 關鍵詞萃取（僅 TextCNN 提供）
- **GET /health** — 服務健康檢查

範例請求：`{"preString": "路燈故障無法修復", "preNum": 3}`

::: notes
/predict 端點接受陳情文字（preString）與候選數量（preNum），回傳 Top-N 局處預測結果及信心值。/tfidf 端點僅 TextCNN 版本提供，用於萃取文字關鍵詞。
:::
