# Backlog — smart-ai

> **唯一狀態來源。** 所有工項的完成 / 取消 / 暫緩都在這裡更新。
> Phase 細節檔（`smart-ai/phase-*.md`）為規格說明，不再負責狀態追蹤。

## 狀態碼

| 符號 | 意義 |
|------|------|
| ⬜ | 待完成 |
| 🔨 | 進行中 |
| ✅ | 完成 |
| 🚫 | 取消 |
| ⏸ | 暫緩（見下方暫緩清單） |

---

## Phase A — 架構分機關

細節規格：[`smart-ai/phase-A-架構分機關.md`](smart-ai/phase-A-架構分機關.md)

| ID | 說明 | 狀態 | 依賴 | 完成日 |
|----|------|------|------|--------|
| A1 | 建立 `smart-ai/` 頂層目錄 + 遷移 `poc-bert/src/` | ✅ | — | 2026-05-13 |
| A2 | `config.py` 改為 YAML loader（向後相容） | ✅ | A1 | 2026-05-13 |
| A3 | 建立 `systems/taitung_bigdata/config.yaml` 規格 | ✅ | A2 | 2026-05-17 |
| A4 | 建立 `systems/chiefmail_back/orgs/{hpa,edu}/` 目錄樹 | ✅ | A3 | 2026-05-17 |
| A5 | ~~預留 lora / llm_base 空目錄~~ | 🚫 | — | — |
| A6 | `core/db_connector.py`（MySQL 實作 + MSSQL abstract） | ✅ | A4 | 2026-05-17 |

---

## Phase B — FastAPI 多系統路由

細節規格：[`smart-ai/phase-B-API.md`](smart-ai/phase-B-API.md)

| ID | 說明 | 狀態 | 依賴 | 完成日 |
|----|------|------|------|--------|
| B1 | `api/main.py`（FastAPI 主程式） | ✅ | A6 | 2026-05-17 |
| B2 | `api/middleware/auth.py`（Token / API Key） | ✅ | B1 | 2026-05-17 |
| B3 | `POST /api/v1/predict`（即時推論，單筆） | ✅ | B2 | 2026-05-17 |
| B4 | `POST /api/v1/batch`（批次推論） | ✅ | B3 | 2026-05-17 |
| B5 | `POST /api/v1/train`（觸發訓練，接收 DB 參數）— 完整實作 | ✅ | B3 | 2026-05-17 |
| B6 | `core/model_manager.py`（Lazy Loading + cache） | ✅ | B1 | 2026-05-17 |

---

## Phase C — 定期訓練排程（選做）

細節規格：[`smart-ai/phase-C-排程.md`](smart-ai/phase-C-排程.md)

| ID | 說明 | 狀態 | 依賴 | 完成日 |
|----|------|------|------|--------|
| C1 | `core/scheduler.py`（APScheduler） | ⬜ | B5 | — |
| C2 | 各系統 / 機關訓練頻率設定 | ⬜ | C1 | — |
| C3 | 訓練完成後自動更新 `current` 軟連結 | ⬜ | C1 | — |

---

## 暫緩項目

| ID | 說明 | 暫緩原因 | 重啟條件 |
|----|------|---------|---------|
| D1 | chiefmail_hpa BERT 基線再訓練 | 精準度非本期重點 | 新架構需實樣資料驗證時 |
| D2 | property_divide 系統 | 待業務需求確認 | 業務需求確認後另起評估 |

---

## 新增工項規則

新增工項時：
1. 給定唯一 ID（延續字母 + 數字，如 `B7`、`D3`）
2. 填寫依賴欄（阻擋項目的 ID）
3. 在對應 Phase 細節檔補充驗收條件
4. 更新 `status.md` 的「當前工項」指針
