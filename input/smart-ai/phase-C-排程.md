# Phase C：定期訓練排程

> **狀態請見 [`../backlog.md`](../backlog.md)**（Phase C 區塊）。本檔為規格說明，不追蹤 checkbox 狀態。
> 預估工期：2-3 天 ｜ 優先序：選做（依賴 Phase A+B）

## 目標

讓 smart-ai 能依排程自動從 DB 拉資料訓練各系統 / 機關模型，並自動更新 `current` 軟連結。

## 工作項目

- [ ] **C1** 實作 `core/scheduler.py`（APScheduler）
- [ ] **C2** 設定各系統 / 機關訓練頻率（每週 / 每月，可依機關覆寫）
- [ ] **C3** 訓練完成後自動更新 `checkpoints/{type}/current` 軟連結指向新版本

## 驗收條件

- **C1**：scheduler 啟動後依設定觸發訓練 job
- **C3**：訓練成功 → `current` 軟連結自動更新；訓練失敗 → 保持原版本，不破壞生產服務

## 注意事項

- RTX 3080 Ti 12GB 是邊界值，建議排程訓練在離峰時段（例如每週日凌晨）
- 訓練期間建議暫停推論服務避免 OOM

## 相關連結

- 規格主檔：[`../smart-ai-plan.md`](../smart-ai-plan.md)
- 前置：[`phase-A-架構分機關.md`](phase-A-架構分機關.md)、[`phase-B-API.md`](phase-B-API.md)
- 上層計畫：[`../../PLAN.md`](../../PLAN.md)
- 開發 skill：`/karpathy-guidelines`
