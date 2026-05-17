# CLAUDE.md — Agent 地圖

> 此檔是 Agent 跨 session 載入時的入口地圖。**不放細節**，只放定位 + 連結。

## 一句話

市民陳情文字自動分類 POC（TextCNN + BERT 雙方案完成），目前重構為 **smart-ai 多系統、多機關平台**。

## 技術棧

- **主環境**：Windows 11 + WSL2 Rocky 9.7 + Docker 29.2.1
- **訓練**：外部 GPU 機器（RTX 3080 Ti 12GB）
- **模型**：BERT（HuggingFace `hfl/chinese-roberta-wwm-ext`）+ TextCNN（PyTorch）
- **API**：FastAPI（Phase B 規劃中）
- **DB**：MySQL（必做）+ MSSQL（Phase A 預留 abstract，待需求）

## 子文件地圖

| 需求 | 路徑 |
|---|---|
| 主計畫 | [`PLAN.md`](PLAN.md) |
| smart-ai 規格 | [`input/smart-ai-plan.md`](input/smart-ai-plan.md) |
| 當前進度 | [`status.md`](status.md) |
| Agent 經驗 | [`learnings.md`](learnings.md) |
| Phase A 細節 | [`input/smart-ai/phase-A-架構分機關.md`](input/smart-ai/phase-A-架構分機關.md) |
| Phase B 細節 | [`input/smart-ai/phase-B-API.md`](input/smart-ai/phase-B-API.md) |
| Phase C 細節 | [`input/smart-ai/phase-C-排程.md`](input/smart-ai/phase-C-排程.md) |
| 暫緩項目 | [`input/smart-ai/deferred.md`](input/smart-ai/deferred.md) |
| 階段一/二 歸檔 | [`docs/plan-archive/phase-1-2.md`](docs/plan-archive/phase-1-2.md) |
| 人類向文件 | [`docs/README.md`](docs/README.md) |
| 簡報素材 | [`notes/README.md`](notes/README.md) |

## 目錄定位

| 路徑 | 用途 |
|---|---|
| `poc/` | TextCNN POC（已完成，保留歷史） |
| `poc-bert/` | BERT POC（已完成，已部署 Port 8081） |
| `smart-ai/` | 重構目標（Phase A 建構中） |
| `docs/` | 人類向技術文件 |
| `input/` | 子計畫提案（待審；審查通過進 PLAN.md） |
| `notes/` | 開發筆記 + pptx-generator 素材源 |

## Session 啟動 SOP

1. 讀 [`status.md`](status.md) 了解上次進度與下次第一步
2. 讀 [`learnings.md`](learnings.md) 索引避免重複錯誤
3. 讀 [`PLAN.md`](PLAN.md) 確認當前 Phase
4. 開工前過 `/skills` 審查（當前實作 skill：`/karpathy-guidelines`）

## 重要規則

1. **繁體中文** — 所有對話、commit、文件、程式碼註解使用繁中
2. **PLAN.md 主、`input/` 子** — 不要自動把 `input/` 拉進 PLAN.md，等使用者選擇
3. **開發前 gate** — 動程式碼之前必過 `/skills`
4. **status.md 入版控** — 不要 gitignore，多人協作共享
5. **精準度非本期重點** — 個別類別（如臺東市公所 recall）不再追，本期焦點是架構
6. **不破壞性 git** — 不執行 `reset --hard`、`push --force` 除非使用者明示
7. **不 commit secrets** — `.env`、credentials.json、access token 一律 ignore
8. **修改超過 5 個檔案前** — 先更新 `PLAN.md` 或對應 phase 檔
9. **session 結束前** — 更新 `status.md`，含「下次第一步」
10. **結果導向** — 不堅持舊技術，選最能達成目標的方案

## 個人記憶位置

`~/.claude/projects/D--daily-records----POC-POC-for-----/memory/`
（已有 4 條：plan workflow、status 版控、臺東 recall 不追、/skills gate）

## 協作節奏

- 每完成一個工作項目 → 更新對應 phase 檔 + `status.md`
- 每個步驟：寫程式 → 解釋原理 → 記錄筆記（`notes/YYYYMMDD_主題.md`）
- 技術說明採「術語 + 白話」對照
