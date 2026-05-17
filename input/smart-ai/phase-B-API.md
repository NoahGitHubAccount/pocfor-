# Phase B：FastAPI 多系統路由

> **狀態請見 [`../backlog.md`](../backlog.md)**（Phase B 區塊）。本檔為規格說明，不追蹤 checkbox 狀態。
> 預估工期：3-5 天 ｜ 優先序：必做（Phase A 之後）

## 目標

建立統一 API 介面，依 `system + org` 路由到對應模型；含模型版本切換與 LRU 快取。

## 工作項目

- [ ] **B1** 實作 `api/main.py`
- [ ] **B2** 實作 `api/middleware/auth.py`（Token 或 API Key）
- [ ] **B3** 實作 `/api/v1/predict`（即時推論，單筆）
- [ ] **B4** 實作 `/api/v1/batch`（批次推論）
- [ ] **B5** 實作 `/api/v1/train`（觸發訓練，接收 DB 參數）
- [ ] **B6** 實作 `core/model_manager.py`（Lazy Loading + LRU Cache + 版本切換）

## API 規格

詳見 [`../smart-ai-plan.md`](../smart-ai-plan.md) 四、API 規格。

## 驗收條件

- **B3**：對 `taitung_bigdata` POST 一筆陳情文字，回傳預測局處 + 信心值
- **B4**：對 `taitung_bigdata` POST 多筆陳情文字，回傳對應預測列表
- **B5**：對 `chiefmail_back/hpa` 觸發訓練 — DB 拉資料 → 訓練 → 模型 checkpoint 寫入 `vYYYYMMDD/` 版本目錄
- **B6**：兩個 system 模型在 12GB VRAM 限制下能正常 LRU 替換，無 OOM

## 模型版本管理

- 版本目錄：`checkpoints/{type}/vYYYYMMDD/`
- 軟連結 `current → vYYYYMMDD` 指向生產版本
- API 請求支援 `version` 參數（選填，預設 `current`）

## 相關連結

- 規格主檔：[`../smart-ai-plan.md`](../smart-ai-plan.md)
- 前置：[`phase-A-架構分機關.md`](phase-A-架構分機關.md)
- 上層計畫：[`../../PLAN.md`](../../PLAN.md)
- 開發 skill：`/karpathy-guidelines`
