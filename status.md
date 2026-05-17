# 智慧分案 POC — 當下狀態

> **最後更新**：2026-05-17
> **進行中**：階段三 — Phase A/B 全部完成，Phase C 暫緩

## 當前工項

**全部完成。** 詳見 [`input/backlog.md`](input/backlog.md)。

| Phase | 狀態 | 說明 |
|-------|------|------|
| A | ✅ | 架構分機關、config.yaml、目錄樹、db_connector |
| B | ✅ | FastAPI 多系統路由、auth、predict/batch/train、model_manager |
| C1/C2 | ⏸ | 排程（程式碼備妥，未接線） |
| C3 | ✅ | current 軟連結（B5 內完成） |

## 下次啟動第一步

**無預排開發工項。** 下次啟動依需求從以下選擇：

1. **L2 端對端驗證** — 在 GPU 機器跑 `docker compose up api`，接真實 DB 測試 `/predict` 與 `/train`
2. **harness 更新** — 依 `docs/工程管理整併方案.md` 更新 CLAUDE.md、learnings.md（可丟給 harness-engineer agent）
3. **Phase C 啟用** — 有排程需求時，在 `api/main.py` 接回 `scheduler.py`，加入 `requirements.txt APScheduler==3.10.4`
4. **新功能提案** — 在 `input/` 建子計畫檔，審查後加入 `input/backlog.md`

## 已完成 checkpoint（本 session）

- **2026-05-17**：
  - Phase A 補完：A2/A3（config.yaml）、A4（chiefmail_back 目錄樹）、A5（取消 lora）、A6（db_connector.py）
  - 工程管理整併：新增 `input/backlog.md`（唯一工項狀態）、`docs/工程管理整併方案.md`
  - Phase B 全部完成：B1–B6（L1 驗收 8/8）、B5 完整實作（DB 抓資料 → 訓練 → versioned checkpoint → current 軟連結）
  - Phase C 暫緩：scheduler.py 備妥未接線
  - 需求情境補充：部署拓樸（dev 多系統 / prod 單系統）記錄至 `input/smart-ai-plan.md`

- **2026-05-13**：Phase A.1 完成、Harness 重整、計畫修訂 v1.3
- **2026-04-20**：BERT val_acc=0.76、TextCNN val_acc=0.62

## Blocker

無。

## Git 狀態

- branch: main
- HEAD: 3c2931d（chore: 暫緩 Phase C 排程）

## 備忘

- `docs/工程管理整併方案.md` → harness-engineer agent 已處理完畢 ✅
- `.claude/hooks/stop-status-snapshot.ps1` 規劃但尚未實作
- 個人記憶位置：`~/.claude/projects/D--daily-records----POC-POC-for-----/memory/`

---

> 歷史 status 條目（超過 1 個月）移至 `status-history/`（首次歸檔時建立目錄）。
