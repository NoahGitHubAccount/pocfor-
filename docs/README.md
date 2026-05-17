# docs/ — 技術文件索引

人類向技術文件。Agent 接手前請先讀 [`../CLAUDE.md`](../CLAUDE.md) + [`../status.md`](../status.md)。

## 文件列表

| 文件                                                       | 用途                              |
| -------------------------------------------------------- | ------------------------------- |
| [`01_自動化訓練文件.md`](01_自動化訓練文件.md)                          | 自動化訓練流程說明                       |
| [`02_預測API文件.md`](02_預測API文件.md)                          | TextCNN / BERT 預測 API 介面與請求格式   |
| [`03_分案系統README.md`](03_分案系統README.md)                    | 分案系統整體說明                        |
| [`04_技術架構文件.md`](04_技術架構文件.md)                            | TextCNN POC 技術架構（階段一移轉文件）       |
| [`GPU訓練操作手冊.md`](GPU訓練操作手冊.md)                            | TextCNN GPU 訓練操作                |
| [`GPU訓練操作手冊_BERT.md`](GPU訓練操作手冊_BERT.md)                  | BERT GPU 訓練操作                   |
| [`plan-archive/phase-1-2.md`](plan-archive/phase-1-2.md) | 階段一/二 計畫歸檔（POC 已完成內容）           |

## 撰寫慣例

- 所有文件使用**繁體中文**
- 每份文件頂部加上「最後更新日期」
- 程式碼範例註明語言與用途
- 截圖放在 `docs/assets/`（若未來新增）
- 超過 200 行的文件考慮拆分

## 與根目錄文件的關係

| 根目錄 | 用途 |
|---|---|
| `README.md` | 入口 + 連結到此目錄 |
| `CLAUDE.md` | Agent 地圖 + 連結到此目錄 |
| `PLAN.md` | 任務計畫（短期，會變動） |
| `docs/` | 永久性參考文件（變動較少） |
| `docs/plan-archive/` | 已完成階段的計畫歸檔 |

## 未來新增

smart-ai 階段三（多 system / 多 org）的技術文件預計於 Phase B 完成後新增：
- `05_smart-ai_架構文件.md`
- `06_smart-ai_API規格.md`
- `07_smart-ai_部署文件.md`
