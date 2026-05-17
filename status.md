# 智慧分案 POC — 當下狀態

> **最後更新**：2026-05-17
> **進行中**：階段三 Phase A 架構分機關（A1+A2+A3 完成，下一步 A4）

## 進行中任務

- [x] [Phase A.1] 建立 `smart-ai/` 頂層目錄 + 遷移 `poc-bert/src/` ✅ 2026-05-13
- [x] [Phase A.2] `config.yaml` 規格化（取代 `config.py`）✅ 2026-05-17（L1 smoke test 全通）
- [ ] [Phase A.3] `systems/{system}/orgs/{org}/` 目錄樹 + LoRA 預留空目錄
- [ ] [Phase A.4] `core/db_connector.py`（MySQL only，MSSQL abstract）

詳見 [`input/smart-ai/phase-A-架構分機關.md`](input/smart-ai/phase-A-架構分機關.md)。

## 已完成 checkpoint

（依時間倒序）

- **2026-05-13**：
  - **Phase A.1 完成**：`smart-ai/src/` 建立，poc-bert/src/ 8 個檔案逐字 copy，diff 空
  - 計畫修訂 v1.3：`input/smart-ai-plan.md` Phase A/B/C 重排，LoRA 暫緩
  - PLAN.md 重組為參照式（細節外推至 `input/smart-ai/*`、`docs/plan-archive/`）
  - Harness 重整：CLAUDE.md / status.md / learnings.md / docs/README / notes/README 對齊現況
- **2026-05-12**：階段一 1.8 技術移轉文件完成（`docs/04_技術架構文件.md`）
- **2026-04-20**：BERT 訓練完成 val_acc=0.76，TextCNN 完成 val_acc=0.62

## 下次啟動第一步

實作 **Phase A.3** — 建立 `systems/{system}/orgs/{org}/` 目錄樹 + LoRA 預留空目錄。

執行前：
1. 過 `/karpathy-guidelines` skill
2. 對照 `phase-A-架構分機關.md` 中的目錄結構產出表
3. 建立 `chiefmail_back/orgs/hpa/` + `checkpoints/bert/` + `checkpoints/lora/`（空目錄用 `.gitkeep`）
4. 建立 `chiefmail_back/llm_base/`（空目錄預留，A5）
5. 驗收：`smart-ai/systems/` 目錄樹符合 phase-A 規格表

## Blocker

無。

## Git 狀態

- branch: main
- HEAD: a244de9（最近一筆，可能已不是最新）

## 備忘

- `.claude/hooks/stop-status-snapshot.ps1` 規劃但尚未實作（CLAUDE 內仍假設它存在 — 待補或刪除假設）
- 個人記憶 4 條已寫入 `~/.claude/projects/D--daily-records----POC-POC-for-----/memory/`

---

> 歷史 status 條目（超過 1 個月）移至 `status-history/`（首次歸檔時建立目錄）。
