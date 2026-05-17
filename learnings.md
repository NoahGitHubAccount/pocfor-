# Agent 經驗檔 — 智慧分案 POC

> **For Agent，非 for human。** 每次 session 開始時 Agent 應主動讀取此檔，避免重複犯錯浪費 token。
> 條目格式：`[YYYY-MM-DD] 情境 → 錯誤 → 根因 → 規避規則`

---

## Top 教訓索引（最重要）

- **不要追個別類別精準度** — 臺東市公所 recall 議題已結案，本期焦點是架構（[2026-05-13]）
- **PLAN.md 與 input/ 是主從** — 不要自動同步（[2026-05-13]）
- **開發前必過 /skills** — 當前 smart-ai 用 `/karpathy-guidelines`（[2026-05-13]）
- **status.md 入版控** — 多人共享，不 gitignore（[2026-05-13]）
- **Docker Desktop 在 Win11 啟動失敗 → WSL2 + Docker Engine** — 不再嘗試 Desktop（[2026-04-15]）

---

## 詳細條目

### 環境 / 工具

#### [2026-04-15] Docker Desktop 在 Windows 11 啟動失敗
- **情境**：嘗試在 Windows 主機跑 Docker Desktop 訓練 BERT
- **錯誤**：Docker Desktop 服務無法啟動，VPN/WSL 衝突
- **根因**：Docker Desktop 與既有 WSL2 distro 設定衝突
- **規避規則**：直接用 **WSL2 + Docker Engine（Rocky9 內裝）**，不裝 Docker Desktop

#### [2026-04-20] 訓練 log 體積大不應入版控
- **情境**：BERT 訓練產生 `bert_train_log.txt` 達 512 KB
- **錯誤**：log 被誤 commit 進版控
- **根因**：`.gitignore` 沒擋 `*_train_log.txt`
- **規避規則**：訓練 log 一律入 `.gitignore`（pattern: `*_train_log.txt`）

---

### 模型 / 訓練

#### [2026-04-18] jieba 斷詞 vs BERT Tokenizer
- **情境**：原 TextCNN 用 jieba 斷詞 + word2vec
- **抉擇**：BERT 方案改用 BERT Tokenizer，自動 subword，不需 jieba
- **規避規則**：BERT-based 模型**不要**再引入 jieba 流程，徒增不一致與維護成本

#### [2026-05-13] 個別類別精準度非優化重點
- **情境**：BERT 在臺東市公所 recall 42%（F1 0.55）
- **錯誤行為（避免）**：曾被視為「下一步優化」並重複提出建議
- **根因**：使用者本意 POC 是實驗，不追個別精準度
- **規避規則**：不要主動提出 class weighting / focal loss / 增補資料建議。除非使用者明示重啟，否則「精準度議題已結案」

---

### 工作流

#### [2026-05-13] PLAN.md 與 input/ 的主從關係
- **情境**：input/ 是子計畫提案區
- **規避規則**：不要把 `input/` 內容自動拉進 PLAN.md。input/ → 使用者審查 → 選擇 → 寫入 PLAN.md。落差是正常的，不必「對齊」

#### [2026-05-13] 開發前須過 /skills 審查
- **情境**：動程式碼 / 建鷹架前
- **規避規則**：先確認用哪個 skill（目前 smart-ai 重構用 `/karpathy-guidelines`），不要裸寫程式

#### [2026-05-13] 計畫文件參照式組織
- **情境**：PLAN.md、`input/smart-ai-plan.md`、status.md 容易越長越大
- **規避規則**：主文件保留骨架 + 狀態欄，**細節外推**到子檔
  - 歷史 → `docs/plan-archive/`
  - phase 細節 → `input/smart-ai/phase-*.md`
  - 過期 status → `status-history/`

#### [2026-05-13] status.md 入版控
- **情境**：多人協作場景
- **規避規則**：status.md 不 gitignore，內容寫得讓下一個接手的人也能看懂，不只 Agent 自己讀

---

### 硬體 / 部署

#### [2026-05-12] RTX 3080 Ti 12GB 是邊界值
- **情境**：規劃同時跑 BERT + LLM 8B 推論
- **風險**：LLM (4-bit) ~5.5GB + BERT ~2GB = 7.5GB；加 LoRA 訓練 ~10GB
- **規避規則**：
  - 訓練期間建議暫停推論服務
  - 模型走 Lazy Loading + LRU Cache
  - 排程訓練放離峰時段（週日凌晨）

---

## 歸檔

超過 6 個月未觸發的條目可移至 `learnings.archive.md`。
