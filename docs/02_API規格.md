# smart-ai — API 規格

> **版本**：v1.0（2026-05-17）
> **Base URL**：`http://<host>:8080`
> **認證**：所有端點（除 /health）需附帶 `X-API-Key` header

---

## 通用規格

| 項目 | 說明 |
|------|------|
| 傳輸格式 | JSON（UTF-8） |
| Content-Type | `application/json` |
| 認證 | `X-API-Key: <key>`（缺少或錯誤 → 401） |
| Port | 8080 |

---

## GET `/health`

不需認證。

```json
{"status": "ok"}
```

---

## POST `/api/v1/predict` — 即時推論（單筆）

### Request

| 欄位 | 型別 | 必填 | 說明 |
|------|------|------|------|
| `system` | string | ✅ | 業務系統識別碼（如 `taitung_bigdata`） |
| `org` | string | 否 | 機關識別碼（如 `hpa`）；無子機關的 system 可省略 |
| `data` | object | ✅ | 輸入欄位（key/value），欄位名稱由 `config.yaml` 的 `input_fields` 定義 |
| `top_n` | integer | 否（預設 3） | 回傳幾個候選結果 |

**範例（含 org）：**

```json
{
  "system": "chiefmail_back",
  "org": "hpa",
  "data": {
    "subject": "檢舉路口違規停車，長期佔用行人穿越道",
    "content": "每日上下班期間均有違停車輛阻礙通行，請派員處理"
  },
  "top_n": 3
}
```

**範例（無子機關）：**

```json
{
  "system": "taitung_bigdata",
  "data": {"subject": "路燈故障已多次通報仍未修復"}
}
```

### Response

```json
{
  "system": "chiefmail_back",
  "org": "hpa",
  "predictions": [
    {"label": "警察局", "confidence": 0.8732},
    {"label": "交通局", "confidence": 0.0891},
    {"label": "建設局", "confidence": 0.0377}
  ]
}
```

**信心值使用建議：**

| 信心值 | 建議處理 |
|--------|---------|
| ≥ 0.80 | 高信心，可直接採用 |
| 0.50 – 0.79 | 中信心，建議人工複核 |
| < 0.50 | 低信心，應由人工判斷 |

---

## POST `/api/v1/batch` — 批次推論

### Request

```json
{
  "system": "chiefmail_back",
  "org": "hpa",
  "items": [
    {"data": {"subject": "路燈故障", "content": "..."}},
    {"data": {"subject": "違規停車", "content": "..."}}
  ],
  "top_n": 3
}
```

### Response

```json
{
  "results": [
    {"predictions": [{"label": "建設局", "confidence": 0.7211}, ...]},
    {"predictions": [{"label": "警察局", "confidence": 0.8901}, ...]}
  ]
}
```

---

## POST `/api/v1/train` — 觸發訓練

從 DB 取資料，訓練後建立版本化 checkpoint 並更新 `current` 軟連結。回傳 **202 Accepted**，訓練在背景非同步執行。

### Request

```json
{
  "system": "chiefmail_back",
  "org": "hpa",
  "db": {
    "host": "192.168.1.10",
    "port": 3306,
    "user": "app_user",
    "database": "petition_db"
  }
}
```

> `password` 從環境變數 `DB_PASSWORD` 讀取，不接受 request 傳入。

### Response（202）

```json
{
  "status": "training_started",
  "version": "v20260517"
}
```

### 訓練流程

```
MySQL 連線 → SELECT label/text（依 config.yaml field_map）
→ shuffle → 切分 80/10/10
→ 寫入 data/ 目錄（TSV 格式）
→ 建立 checkpoints/vYYYYMMDD/
→ BERT fine-tune
→ 更新 current 軟連結
```

---

## 錯誤碼

| HTTP | 情境 |
|------|------|
| 400 | 請求格式錯誤或必填欄位缺少 |
| 401 | `X-API-Key` 缺少或錯誤 |
| 404 | system/org 的 config.yaml 找不到 |
| 500 | 伺服器未設定 `SMART_AI_API_KEY`，或 DB 連線失敗 |
| 503 | 模型未載入（`current` 軟連結不存在） |
