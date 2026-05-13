# 智慧分案POC

> 基於 NLP 與 BERT 的法律案件自動分案系統

## 技術棧
Python, FastAPI, BERT, Word2Vec, Docker

## 語言規則
- 所有對話、commit、文件、程式碼註解使用**繁體中文**

## 行為約束
- 不得執行破壞性 git 指令（`reset --hard`、`push --force`）除非使用者明示
- 不得 commit secrets（.env、credentials.json、access token）
- 修改超過 5 個檔案前，先更新 `plan.md`
- session 結束前更新 `status.md`，含「下次第一步」

## 子文件地圖

| 需求 | 路徑 |
|---|---|
| 詳細架構 | `docs/architecture.md` |
| 安裝步驟 | `docs/install.md` |
| 使用方式 | `docs/usage.md` |
| 過去錯誤 | `learnings.md` |
| 任務計畫 | `plan.md` |
| 當前進度 | `status.md` |
| 設計決策 | `docs/adr/` |
| 成果素材 | `notes/` |

## Session 啟動 SOP
1. 讀 `status.md` 了解上次進度
2. 讀 `learnings.md` 索引避免重複錯誤
3. 讀 `plan.md` 確認當前 Phase
4. 開工
