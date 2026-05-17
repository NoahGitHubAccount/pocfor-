# docs/ — 技術文件索引

人類向技術文件。Agent 接手前請先讀 [`../CLAUDE.md`](../CLAUDE.md) + [`../status.md`](../status.md)。

## 文件列表

### 現行版本（smart-ai）

| 文件 | 用途 |
|------|------|
| [`01_技術架構.md`](01_技術架構.md) | 系統演進、smart-ai 目錄結構、設定階層、部署、模型評估 |
| [`02_API規格.md`](02_API規格.md) | /predict、/batch、/train 端點規格與範例 |
| [`03_新專案導入.md`](03_新專案導入.md) | 新客戶導入指引（標籤設計、資料準備、config.yaml、驗收） |
| [`GPU訓練操作手冊_BERT.md`](GPU訓練操作手冊_BERT.md) | 手動 GPU 訓練操作（BERT；smart-ai 優先用 /train API） |
| [`工程管理整併方案.md`](工程管理整併方案.md) | backlog.md 作為唯一工項狀態來源的設計說明（內部） |

### 歸檔（`plan-archive/`）

| 文件 | 說明 |
|------|------|
| `phase-1-2.md` | 階段一/二計畫歸檔 |
| `poc_textcnn_*.md` | TextCNN 相關文件（已驗證，不導入 prod） |

## 與根目錄文件的關係

| 根目錄 | 用途 |
|--------|------|
| `README.md` | 入口 + 連結到此目錄 |
| `CLAUDE.md` | Agent 地圖 + 連結到此目錄 |
| `PLAN.md` | 任務計畫（短期，會變動） |
| `docs/` | 永久性參考文件（變動較少） |
| `docs/plan-archive/` | 已完成或封存的歷史文件 |

## 未來新增

Phase C 啟用後補充：
- `04_排程設定.md`（APScheduler 設定、各 system 訓練頻率）
