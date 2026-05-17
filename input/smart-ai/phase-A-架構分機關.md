# Phase A：架構分機關

> **狀態請見 [`../backlog.md`](../backlog.md)**（Phase A 區塊）。本檔為規格說明，不追蹤 checkbox 狀態。
> 預估工期：3-5 天 ｜ 優先序：必做（先於 Phase B）

## 目標

建立可分系統、分機關的目錄結構與設定規格，並提供 DB 連線統一介面。**僅搬遷與外推設定，不動現有模型邏輯。**

## 工作項目

- [x] **A1** 建立 `smart-ai/` 頂層目錄（與 `poc/`、`poc-bert/` 平行）— 2026-05-13
- [x] **A2** 遷移 `poc-bert/src/` 至 `smart-ai/src/`（8 檔逐字 copy，diff 空）— 2026-05-13
- [x] **A3** 設計 `config.yaml` 規格，取代既有 `config.py` — 2026-05-17
- [ ] **A4** 建立 `systems/taitung_bigdata/` 與 `systems/chiefmail_back/orgs/{org}/` 目錄樹
- [~] **A5** ~~預留 `llm_base/`、`orgs/{org}/checkpoints/lora/` 空目錄~~ — 🚫 取消，LoRA 不在本期範圍
- [ ] **A6** 實作 `core/db_connector.py`（**僅 MySQL**；MSSQL 預留 abstract class 不寫實作）

## 已決議假設（karpathy rule 1）

| 假設 | 決議 |
|---|---|
| `smart-ai/` 位置 | 專案根目錄，與 `poc/`、`poc-bert/` 平行，**不取代** |
| MSSQL 連線 | 暫不實作，預留 abstract class；等 chiefmail_back 接入再寫 |
| 模型邏輯 | **不改**，純搬遷 + 設定外推 |
| 密碼管理 | DB 連線資訊走 `.env`，不寫死 yaml |

## 驗收條件（karpathy rule 4）

- **A1-A2**：`smart-ai/` 目錄樹存在；`smart-ai/src/predict.py` 可載入 `poc-bert/checkpoints/bert-model` 並等效呼叫
- **A3**：`config.yaml` 載入後 `train.py` 可跑 dry-run 一個 epoch
- **A6**：`db_connector.connect()` 對 MySQL 可建立連線並執行 `SELECT 1`

## 目錄結構（產出）

```
smart-ai/
├── api/                     # Phase B 才實作，A 階段建空目錄
├── core/
│   ├── db_connector.py      # A6
│   ├── model_manager.py     # Phase B
│   └── scheduler.py         # Phase C
├── systems/
│   ├── taitung_bigdata/
│   │   ├── config.yaml
│   │   └── checkpoints/bert/
│   ├── chiefmail_back/
│   │   ├── config.yaml
│   │   └── orgs/
│   │       └── hpa/
│   │           ├── config.yaml
│   │           └── checkpoints/
│   │               └── bert/
│   └── property_divide/     # 待需求
└── src/                     # A2 從 poc-bert/src/ 遷移
    ├── train_bert.py
    ├── predict.py
    └── data_loader.py
```

## 相關連結

- 規格主檔：[`../smart-ai-plan.md`](../smart-ai-plan.md)
- 上層計畫：[`../../PLAN.md`](../../PLAN.md)
- 開發 skill：`/karpathy-guidelines`
