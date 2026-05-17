# smart-ai 系統架構修改計劃

> 版本：v1.3  
> 日期：2026-05-13  
> 環境：Linux / RTX 3080 Ti 12GB / MySQL & MSSQL
> 
> **v1.3 變更**：依使用者決議重排優先序 — 先做「架構分機關」、再做「API 開發」、排程訓練選做；LLM LoRA 暫緩評估，property_divide 待業務需求。精準度（含臺東市公所 recall）非本期重點。

---

## 零、部署情境（架構決策前提）

> 本節為 2026-05-17 需求訪談補充，影響 API 設計與模型管理策略。

### 環境拓樸

| 環境 | 說明 | 系統數量 |
|------|------|---------|
| **dev** | 所有系統在同一台機器，驗證多系統並存 | 多個（taitung_bigdata + chiefmail_back + ...） |
| **test / stage** | 可能依機關分配到獨立機器 | 視測試需求 |
| **prod（客戶端）** | 每個客戶有自己的機器，**僅啟用一個系統** | 1 |

### 關鍵決策影響

1. **三個系統是三個獨立產品**，服務三個不同客戶，prod 環境彼此完全隔離，不會同時在線。
2. **API 需支援兩種部署模式**：
   - dev：多系統並存，`system + org` 路由到對應模型
   - prod：單系統啟動，`system + org` 由啟動設定決定，API 介面相同
3. **模型 LRU 快取**只在 dev/test 環境有意義；prod 只載入一個模型，不需要 LRU 替換。
4. **API 僅服務後台 AP 系統**，兩台均在同一內網，不對外部提供服務，API Key 驗證已足夠。

---

## 一、需求摘要

### 三大系統

| 系統代號 | 來源欄位 | 輸出需求 | 模型類型 |
|---------|---------|---------|---------|
| taitung_bigdata | 陳情文字 | 局處分類 | BERT 分類 |
| chiefmail_back | 主旨、內容 | 指派單位 / 建議類別 / 建議回覆 | BERT 分類 + LLM 生成 |
| property_divide | 異動類別 | 待定 | 待定 |

### chiefmail_back 多機關需求

- 相同輸入欄位（主旨、內容），不同機關資料
- 各機關標籤獨立（BERT 各自訓練）
- 回覆風格可能不同（LLM 共用底座 + 各機關獨立 LoRA）

### AP 與 smart-ai 互動方式

- 即時推論：AP 拋轉單筆資料，即時取得預測結果
- 批次推論：定時拋轉多筆資料
- 訓練資料來源：smart-ai 連線 DB（MySQL 或 MSSQL）定期抓取

---

## 二、整體流程

```
AP 系統
  ├── 即時推論 → POST /api/v1/predict → 回傳結果
  ├── 批次推論 → POST /api/v1/batch  → 回傳結果
  └── 資料存入 DB

smart-ai（定期）
  └── 連線 DB → 抓取訓練資料 → 批次訓練 → 更新模型
```

---

## 三、系統架構

```
smart-ai/
├── api/
│   ├── main.py                  ← FastAPI 主程式
│   ├── routers/
│   │   ├── predict.py           ← 即時推論
│   │   ├── batch.py             ← 批次推論
│   │   └── train.py             ← 觸發訓練
│   └── middleware/
│       └── auth.py              ← API 驗證
│
├── core/
│   ├── model_manager.py         ← 模型載入/切換/快取
│   ├── db_connector.py          ← MySQL/MSSQL 統一介面
│   └── scheduler.py             ← 定期訓練排程
│
├── systems/
│   ├── taitung_bigdata/
│   │   ├── config.yaml
│   │   └── checkpoints/
│   │       └── bert/            ← 引入版本化 (v20260512/)
│   │
│   ├── chiefmail_back/
│   │   ├── config.yaml
│   │   ├── llm_base/            ← 共用 LLM 底座（Llama-3-TAIDE-8B 或 Breeze-7B）
│   │   └── orgs/
│   │       ├── hpa/
│   │       │   ├── config.yaml
│   │       │   └── checkpoints/
│   │       │       ├── bert/    ← 引入版本化 (v20260512/)
│   │       │       └── lora/    ← 引入版本化 (v20260512/)
│   │       └── edu/
│   │           ├── config.yaml
│   │           └── checkpoints/
│   │               ├── bert/
│   │               └── lora/
│   │
│   └── property_divide/
│       ├── config.yaml
│       └── checkpoints/
│
└── src/
    ├── train_bert.py
    ├── train_llm.py
    ├── predict.py
    └── data_loader.py
```

---

## 四、API 規格

### 即時推論
```
POST /api/v1/predict
Content-Type: application/json

Request：
{
  "system": "chiefmail_back",
  "org": "hpa",
  "version": "v20260512",  # (選填) 預設為 current
  "data": {
    "subject": "HPV疫苗詢問",
    "content": "請問哪裡可以施打？"
  }
}

Response：
{
  "unit": "癌症防治組",
  "category": "諮詢類",
  "reply": "您好，HPV疫苗可至本局合約院所施打..."
}
```

### 批次推論
```
POST /api/v1/batch
{
  "system": "chiefmail_back",
  "org": "hpa",
  "version": "current",
  "data": [
    {"subject": "...", "content": "..."},
    {"subject": "...", "content": "..."}
  ]
}
```

### 觸發訓練
```
POST /api/v1/train
{
  "system": "chiefmail_back",
  "org": "hpa",
  "db": {
    "type": "mysql",
    "host": "192.168.1.1",
    "database": "chiefmail",
    "table": "mail_records"
  }
}
```

---

## 五、config.yaml 規格

```yaml
system: chiefmail_back
org: hpa
org_name: 衛生局

input_fields:
  - subject
  - content

outputs:
  unit:     true
  category: true
  reply:    true

num_classes: 10

db:
  type: mysql
  table: mail_records
  field_map:
    subject: mail_subject
    content: mail_content
    label_unit: dept_code
    label_category: category_id
    reply: std_reply

epochs: 5
batch_size: 16
max_length: 512
learning_rate: 2e-5
```

---

## 六、可行性評估

### 6.1 硬體可行性

| 項目 | 需求 | 現有 | 評估 |
|------|------|------|------|
| GPU VRAM | 10-12GB | RTX 3080 Ti 12GB | ✅ 剛好足夠 |
| LLM 底座（8B 4-bit） | ~5.5GB | 12GB | ✅ 可行（Llama-3-TAIDE 或 Breeze） |
| BERT 分類 | ~2GB | 12GB | ✅ 可行 |
| LLM + BERT 同時載入 | ~7.5GB | 12GB | ⚠️ 需要測試 |
| LoRA 訓練 | ~10GB | 12GB | ✅ 可行（需關閉推論服務） |

> ⚠️ **注意：** 12GB 是邊界值，訓練期間建議關閉推論服務，避免 OOM（記憶體不足）。

### 6.2 技術可行性

| 項目 | 技術方案 | 可行性 | 備註 |
|------|---------|--------|------|
| BERT 多系統分類 | HuggingFace Transformers | ✅ 高 | 已驗證 |
| LLM 生成（離線） | Llama-3-TAIDE/Breeze + LoRA | ✅ 高 | 台灣優化，非中國模型 |
| MySQL 連線 | PyMySQL / SQLAlchemy | ✅ 高 | 成熟方案 |
| MSSQL 連線 | pymssql / pyodbc | ✅ 高 | 需安裝驅動 |
| 多機關路由 | FastAPI + config.yaml | ✅ 高 | 設計清晰 |
| 定期訓練排程 | APScheduler / Cron | ✅ 高 | 標準方案 |
| 即時 + 批次推論 | FastAPI 非同步 | ✅ 高 | 需設計 queue |

### 6.3 資料可行性

| 項目            | 建議量            | 評估            |
| ------------- | -------------- | ------------- |
| BERT 分類訓練資料   | 每類 100 筆以上     | 需確認各機關資料量     |
| LLM LoRA 訓練資料 | 500 筆以上        | 已有 2000 筆以上 ✅ |
| 資料格式統一        | Tab 分隔 / JSONL | 需依 DB 欄位對應設計  |

---

## 七、修改計劃（v1.3 重排）

> **本期執行範圍**：Phase A 架構分機關 → Phase B API 開發 → Phase C 排程訓練（選做）
> 細節已外推至獨立檔案以利維護。

| Phase | 內容 | 預估 | 細節檔 |
|---|---|---|---|
| A | 架構分機關 | 3-5 天 | [`smart-ai/phase-A-架構分機關.md`](smart-ai/phase-A-架構分機關.md) |
| B | FastAPI 多路由 | 3-5 天 | [`smart-ai/phase-B-API.md`](smart-ai/phase-B-API.md) |
| C | 定期訓練排程（選做） | 2-3 天 | [`smart-ai/phase-C-排程.md`](smart-ai/phase-C-排程.md) |
| — | 暫緩項目（LoRA、property_divide、基線再訓練） | — | [`smart-ai/deferred.md`](smart-ai/deferred.md) |

---

## 八、建議

### 8.1 短期建議

1. **先跑完 BERT 分類**，確認基礎分類效果，再擴展架構，避免架構還沒穩定就引入 LLM 的複雜度。

2. **LLM 生成建議分兩步驟驗證**：
   - 第一步先用 API 模式（如 OpenAI 格式的本地 LLM）驗證 Prompt 效果
   - 確認回覆品質後，再進行 LoRA Fine-tune

3. **訓練與推論分開部署**：RTX 3080 Ti 12GB 是邊界值，建議訓練期間暫停推論服務，或規劃訓練時段（如每週日凌晨）。

### 8.2 中期建議

4. **DB 連線資訊不應寫死在 config.yaml**，建議使用環境變數（`.env`）管理，避免密碼洩漏：
   ```
   DB_HOST=192.168.1.1
   DB_PASSWORD=xxxxxxxx
   ```

5. **模型版本管理與部署彈性**：
   - 採用目錄版本化：`checkpoints/{type}/vYYYYMMDD/`。
   - 使用軟連結：`current -> vYYYYMMDD` 指向生產環境模型，實現秒級回滾。
   - API 請求支援 `version` 參數，方便 A/B Test 或開發測試。

6. **VRAM 資源動態管理 (RTX 3080 Ti 12GB)**：
   - **Lazy Loading**：API 啟動時不預載所有機關模型，僅在首次請求時載入。
   - **LRU Cache**：實作模型快取機制，當多機關切換導致 VRAM 不足時，自動釋放最久未使用的模型。
   - **預測信心度 (Confidence Score)**：當分類信心度低於門檻時，回傳「需人工審核」，避免誤導。

### 8.3 長期建議

7. **建立資料回饋機制**：AP 系統讓使用者確認或修正預測結果，修正資料自動回存 DB，作為下次訓練的優質樣本。

8. **property_divide 系統**待需求確認後，評估是否與 chiefmail_back 共用 LLM 底座，進一步節省資源。

---

## 九、風險與對策

| 風險 | 可能性 | 影響 | 對策 |
|------|--------|------|------|
| 12GB VRAM 不足 | 中 | 高 | 量化、訓練推論分開、LRU Cache 釋放 VRAM |
| 各機關資料量不足 | 中 | 高 | 資料增強、調低分類數、合併相似標籤 |
| DB 連線異常 | 低 | 中 | 加入 retry 機制、離線備用資料集 |
| LLM 回覆品質不佳 | 中 | 中 | 加強 Prompt 設計、增加訓練資料量 |
| 多機關模型混用 | 低 | 高 | 嚴格依 system+org 路由，單元測試覆蓋 |

---

*本文件依現有需求討論產生，架構細節可依實際開發情況調整。*
