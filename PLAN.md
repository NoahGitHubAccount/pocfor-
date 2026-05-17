# 智慧分案 POC 重建計畫

> **最後更新**：2026-05-13
> **當前焦點**：階段三 Phase A — smart-ai 架構分機關（未開工）
> **下次第一步**：見 [`status.md`](status.md)

---

## 路線圖

| 階段 | 焦點                | 狀態     | 詳細                                                                |
| -- | ----------------- | ------ | ----------------------------------------------------------------- |
| 一  | TextCNN POC       | ✅ 完成   | [`docs/plan-archive/phase-1-2.md`](docs/plan-archive/phase-1-2.md) |
| 二  | BERT POC          | ✅ 完成   | [`docs/plan-archive/phase-1-2.md`](docs/plan-archive/phase-1-2.md) |
| 三  | smart-ai 多系統、多機關  | 🔨 進行中 | 見下方                                                              |

---

## 階段三：smart-ai 架構重建

> **規格主檔**：[`input/smart-ai-plan.md`](input/smart-ai-plan.md)（v1.3，初版計畫存檔）
> **工項追蹤**：[`input/backlog.md`](input/backlog.md)（唯一狀態來源，Issue 管理）
> **暫緩項目**：[`input/smart-ai/deferred.md`](input/smart-ai/deferred.md)
> **開發前 gate**：實作前須先過 `/skills` 審查（當前 skill：`/karpathy-guidelines`）

| Phase | 內容                  | 狀態      | 細節                                                                              |
| ----- | ------------------- | ------- | ------------------------------------------------------------------------------- |
| A     | 架構分機關（必做、優先）        | 🔨 待完成  | [`input/smart-ai/phase-A-架構分機關.md`](input/smart-ai/phase-A-架構分機關.md)             |
| B     | FastAPI 多路由（必做）     | 🔨 待完成  | [`input/smart-ai/phase-B-API.md`](input/smart-ai/phase-B-API.md)                 |
| C     | 定期訓練排程（選做）          | 🔨 待完成  | [`input/smart-ai/phase-C-排程.md`](input/smart-ai/phase-C-排程.md)                   |

---

## 執行環境

| 項目 | 現況 |
|---|---|
| 主環境 | Windows 11 + WSL2 Rocky 9.7 + Docker 29.2.1 |
| 訓練 | 外部 GPU 機器（RTX 3080 Ti 12GB） |
| 推論 | Docker 容器；本機不裝 Python 套件 |
| 現有容器 | `n8n_poc`、`customer-service-classifier` |

詳細執行環境參數見 [`docs/plan-archive/phase-1-2.md`](docs/plan-archive/phase-1-2.md)（POC 階段建立）。

---

## 協作約定

- 每完成一個工作項目 → 更新對應 phase 檔案狀態 + [`status.md`](status.md)
- 每個步驟：寫程式 → 解釋原理 → 記錄筆記（[`notes/`](notes/)）
- 技術說明採「術語 + 白話」對照
- 安裝任何軟體前先確認是否已存在
- **結果導向**：選最能達成目標的方案，不堅持舊技術

---

## 進入點

| 需求 | 路徑 |
|---|---|
| Agent 地圖 | [`CLAUDE.md`](CLAUDE.md) |
| 當下進度 | [`status.md`](status.md) |
| Agent 經驗 | [`learnings.md`](learnings.md) |
| 人類向文件 | [`docs/README.md`](docs/README.md) |
| 開發筆記 / 簡報素材 | [`notes/README.md`](notes/README.md) |
