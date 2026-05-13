# 智慧分案POC Notes

> 此目錄為 **`pptx-generator` skill 的素材源**。每當專案累積出可分享的成果時，在此建立一份 md 檔案，再呼叫 pptx-generator 產出簡報。

## 命名規則

`YYYYMMDD-topic-slug.md`

範例：`20260505-launch-summary.md`

## 必含 Schema

每份素材檔頭部必含 YAML frontmatter：

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
4. `## 4. 套件依賴`（package.json / requirements.txt 摘錄）
5. `## 5. Skill 依賴`（用到哪些 .agent/skills 或 .claude/skills）
6. `## 6. 成效數據`（量化結果、節省時間 / 成本）
7. `## 7. 關鍵截圖`（指向 `notes/assets/` 下圖片）
8. `## 8. 後續展望`

## 與 pptx-generator 對接

```
notes/20260505-launch.md
       ↓
   pptx-generator
       ↓
Pandoc Markdown
       ↓
     PPTX
```

詳細對接契約見 `~/.claude/skills/harness-engineer/references/pptx-handoff.md`。

## 子目錄

- `assets/`：截圖、圖表（PNG / JPG）
- `archive/`：超過一年的舊素材歸檔

## 反模式

- ❌ 在此放零散筆記、未結構化（pptx-generator 讀不到）
- ❌ 跳過 frontmatter（沒有標題頁）
- ❌ 截圖用外部 URL（離線無法生成）
- ❌ 一份超過 1000 字（簡報塞不下）
