# notes/ — 開發筆記 + 簡報素材

> 此目錄為 **`pptx-generator` skill 的素材源**，同時也是日常開發筆記區。

## 用途

1. **開發筆記**：以日期命名，記錄學習路線、技術決策、訓練數據觀察（自由格式）
2. **簡報素材**：累積出可分享的成果時建立一份 md，再呼叫 pptx-generator 產出簡報（需含 schema）
3. **`chat-logs/`**：SessionEnd hook 自動存放對話紀錄（已被 gitignore）

## 命名規則

| 類型 | 格式 | 範例 |
|---|---|---|
| 一般筆記 | `YYYYMMDD_主題.md` | `20260420_訓練數據報告.md` |
| 同日多份 | `YYYYMMDD_HHMM_主題.md` | `20260412_1058_筆記_xxx.md` |
| 簡報素材 | `YYYYMMDD-topic-slug.md` | `20260505-launch-summary.md` |

## 簡報素材 Schema（pptx-generator 銜接）

每份簡報素材檔需含 YAML frontmatter：

```yaml
---
title: 簡報主題
tagline: 一句話定位
date: YYYY-MM-DD
author: 張捷
duration_minutes: 15
audience: 內部技術分享 / 客戶提案 / 主管彙報
---
```

主體必含區塊（依序）：

1. `## 1. 專案簡介`
2. `## 2. 初始 Prompt`（驅動專案的最初指令，原文引用）
3. `## 3. 過程 Prompt 摘要`（重要轉折點，最多 5 個）
4. `## 4. 套件依賴`（`requirements.txt` 摘錄）
5. `## 5. Skill 依賴`（用到的 plugin skills 名稱）
6. `## 6. 成效數據`（量化結果、節省時間 / 成本）
7. `## 7. 關鍵截圖`（指向 `notes/assets/` 下圖片）
8. `## 8. 後續展望`

詳細對接契約見 `~/.claude/skills/harness-engineer/references/pptx-handoff.md`。

## 子目錄

- `assets/`：截圖、圖表（PNG / JPG），未來建立
- `chat-logs/`：SessionEnd hook 對話紀錄存檔（gitignored）
- `archive/`：超過一年的舊素材歸檔，未來建立

## 現有筆記

- `20260412_*`（4 份） — POC 規劃期筆記，含 AI 學習路線、工程化協作願景、整併版
- `20260420_訓練數據報告.md` — TextCNN / BERT 訓練成果（已是簡報素材級）
- `chat-logs/` — 自動存檔

## 反模式

- ❌ 零散筆記未結構化（pptx-generator 讀不到）
- ❌ 簡報素材跳過 frontmatter（沒有標題頁）
- ❌ 截圖用外部 URL（離線無法生成）
- ❌ 一份簡報素材超過 1000 字（簡報塞不下）
