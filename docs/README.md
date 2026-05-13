# 智慧分案POC 文件

人類可閱讀的專案文件目錄。

## 文件索引

| 文件 | 說明 |
|---|---|
| [architecture.md](./architecture.md) | 系統架構與模組劃分 |
| [install.md](./install.md) | 詳細安裝步驟 |
| [usage.md](./usage.md) | 詳細使用方式 |
| [api.md](./api.md) | API 規格（如適用） |
| [deployment.md](./deployment.md) | 部署指南 |
| [troubleshooting.md](./troubleshooting.md) | 疑難排解 |
| [adr/](./adr/) | 架構決策記錄 |

## 撰寫慣例

- 所有文件使用**繁體中文**
- 每份文件頂部加上**最後更新日期**
- 程式碼範例註明語言與用途
- 截圖放在 `docs/assets/`
- 超過 200 行的文件考慮拆分

## 與根目錄文件的關係

| 根目錄 | docs/ |
|---|---|
| README.md | 入口 + 連結到此 |
| CLAUDE.md | Agent 地圖 + 連結到此 |
| plan.md | 任務計畫（短期） |
| 此處 | 永久性參考文件 |
