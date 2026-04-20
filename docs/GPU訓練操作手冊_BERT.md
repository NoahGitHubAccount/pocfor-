# GPU 機器訓練操作手冊（BERT 版）

> BERT 訓練比 TextCNN 更簡單：不需要 jieba 斷詞、不需要 word2vec。
> 預估 GPU 訓練時間：15-30 分鐘（含模型下載）

---

## 一、前置確認

和 TextCNN 版相同，需確認：
- `nvidia-smi` 可看到 GPU
- Docker 已安裝（Docker Desktop 或 WSL2 + Docker Engine）

詳見 `GPU訓練操作手冊.md` 第一節，此處不重複。
**Docker Desktop 無法啟動者請使用方法 B（WSL2 + Docker Engine）。**

---

## 二、搬移檔案到 GPU 機器

### 需要複製的內容

| 來源（你的筆電） | 說明 |
|----------------|------|
| `poc-bert/` 整個資料夾 | 程式碼、Dockerfile |
| `newData_1000_title/` 裡的三個檔案 | 訓練資料 |

只需要這三個訓練資料檔：
- `cnews.train.txt`
- `cnews.val.txt`
- `cnews.test.txt`

（不需要 vocab.txt、vector_word.npz，BERT 不用這些）

### 直接複製（GPU 機器是 Windows）

在 GPU 機器上建立工作目錄：

```powershell
mkdir C:\smart-case-bert\poc-bert
mkdir C:\smart-case-bert\data
```

用檔案總管或隨身碟，將以下內容複製到 GPU 機器：

| 來源 | 複製到 |
|------|--------|
| `poc-bert/` 整個資料夾 | `C:\smart-case-bert\poc-bert\` |
| `newData_1000_title/` 內的 `cnews.*.txt` | `C:\smart-case-bert\data\` |

目標結構：

```
C:\smart-case-bert\
├── poc-bert\
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   ├── checkpoints\
│   └── src\
└── data\
    ├── cnews.train.txt
    ├── cnews.val.txt
    └── cnews.test.txt
```

> **使用方法 B（WSL2 Ubuntu）者**：Windows 路徑對應 WSL2 路徑如下：
> `C:\smart-case-bert\` → `/mnt/c/smart-case-bert/`

---

## 三、修改 Dockerfile — GPU 版 PyTorch

找到 `poc-bert/Dockerfile` 內這行並修改：

```dockerfile
# 原本（CPU 版）
--index-url https://download.pytorch.org/whl/cpu

# 改為對應 CUDA 版本（擇一）
--index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1 / 12.6 推薦
--index-url https://download.pytorch.org/whl/cu124   # CUDA 12.4
```

> 用 `nvidia-smi` 確認 CUDA Version，選最接近的。cu121 可相容 CUDA 12.x 全系列。

**方法 A（PowerShell，記事本或 VS Code 開啟編輯）：**

```powershell
cd C:\smart-case-bert\poc-bert
notepad Dockerfile
```

**方法 B（WSL2 Ubuntu，sed 直接替換）：**

```bash
sed -i 's/whl\/cpu/whl\/cu121/g' /mnt/c/smart-case-bert/poc-bert/Dockerfile
```

---

## 四、Build Image

**方法 A（PowerShell）：**

```powershell
cd C:\smart-case-bert\poc-bert
docker build -t smart-case-bert:gpu .
```

**方法 B（WSL2 Ubuntu）：**

```bash
cd /mnt/c/smart-case-bert/poc-bert
docker build -t smart-case-bert:gpu .
```

> 第一次 build 需下載 PyTorch GPU + transformers，約需 5-10 分鐘（依網速）。

---

## 五、執行訓練

**方法 A（PowerShell）：**

```powershell
docker run --rm --gpus all `
  -e PYTHONUNBUFFERED=1 `
  -v C:\smart-case-bert\data:/app/data:ro `
  -v C:\smart-case-bert\poc-bert\checkpoints:/app/checkpoints `
  -v C:\smart-case-bert\poc-bert\src:/app/src `
  smart-case-bert:gpu `
  python src/train.py 2>&1 | Tee-Object -FilePath C:\smart-case-bert\bert_train_log.txt
```

**方法 B（WSL2 Ubuntu）：**

```bash
docker run --rm --gpus all \
  -e PYTHONUNBUFFERED=1 \
  -v /mnt/c/smart-case-bert/data:/app/data:ro \
  -v /mnt/c/smart-case-bert/poc-bert/checkpoints:/app/checkpoints \
  -v /mnt/c/smart-case-bert/poc-bert/src:/app/src \
  smart-case-bert:gpu \
  python src/train.py 2>&1 | tee /mnt/c/smart-case-bert/bert_train_log.txt
```

> **注意**：第一次執行會自動從 HuggingFace Hub 下載 `hfl/chinese-roberta-wwm-ext`（約 400MB），
> 下載完畢後會自動開始訓練。

### 預期輸出

```
[train] 使用裝置：cuda
[train] 載入 tokenizer：hfl/chinese-roberta-wwm-ext
[train] 準備訓練資料...
[data_loader] Tokenizing 23329 筆（cnews.train.txt）...
[train] 準備驗證資料...
[data_loader] Tokenizing 2916 筆（cnews.val.txt）...
[model] BERT 模型載入完成：hfl/chinese-roberta-wwm-ext
[model] 參數量：102,273,545

[train] 開始訓練，5 epochs
------------------------------------------------------------
{'loss': 1.234, 'learning_rate': 1.5e-05, 'epoch': 0.34}
{'loss': 0.567, 'learning_rate': 1.2e-05, 'epoch': 0.68}
...
{'eval_loss': 0.xxx, 'eval_accuracy': 0.xxxx, 'epoch': 1.0}
...
{'eval_loss': 0.xxx, 'eval_accuracy': 0.xxxx, 'epoch': 5.0}

[train] 最終驗證集評估：
  val_loss     = x.xxxx
  val_accuracy = x.xxxx

[train] 分類報告：
                    precision    recall  f1-score   support
  交通及觀光發展處      x.xx      x.xx      x.xx       xxx
  ...

[train] 模型已存至：/app/checkpoints/bert-model
```

---

## 六、確認訓練成果

**方法 A（PowerShell）：**
```powershell
dir C:\smart-case-bert\poc-bert\checkpoints\
```

**方法 B（WSL2 Ubuntu）：**
```bash
ls -lh /mnt/c/smart-case-bert/poc-bert/checkpoints/
```

預期看到：

```
labels.txt                        （標籤對照表）
bert-model/                       （目錄）
  ├── config.json                 （模型配置）
  ├── model.safetensors           （模型權重，約 400MB）
  ├── tokenizer.json              （tokenizer 配置）
  ├── tokenizer_config.json
  ├── special_tokens_map.json
  └── vocab.txt                   （BERT 詞彙表）
```

---

## 七、把成果帶回筆電

### 需要複製回來的檔案

| 檔案 / 目錄 | 說明 |
|-------------|------|
| `checkpoints/labels.txt` | **必要**，標籤對照表 |
| `checkpoints/bert-model/` 整個目錄 | **必要**，BERT 模型（約 400MB） |
| `bert_train_log.txt` | **必要**，訓練紀錄 |

### 複製回筆電

直接用隨身碟或網路芳鄰複製：

| GPU 機器上的路徑 | 複製到筆電 |
|-----------------|-----------|
| `C:\smart-case-bert\poc-bert\checkpoints\labels.txt` | `D:\POC_for_智慧分案\poc-bert\checkpoints\` |
| `C:\smart-case-bert\poc-bert\checkpoints\bert-model\`（整個目錄） | `D:\POC_for_智慧分案\poc-bert\checkpoints\` |
| `C:\smart-case-bert\bert_train_log.txt` | `D:\POC_for_智慧分案\` |

> **方法 B 用戶**：檔案存在 `/mnt/c/smart-case-bert/` 下，對應 Windows 路徑 `C:\smart-case-bert\`，直接用檔案總管複製即可。

---

## 八、回報給 Claude 確認

回到 Claude Code 對話，提供：

1. **貼上 `bert_train_log.txt` 最後 40 行**（或告訴我最終 val_accuracy）
2. 確認 `checkpoints/bert-model/` 目錄內有 `model.safetensors`
3. Claude 會進行：
   - 用測試集跑準確度
   - 用範例文字測試預測
   - 啟動 API 服務驗證
   - 更新 PLAN.md

---

## 九、常見問題

**Q：CUDA out of memory？**
在 `config.py` 中調小 `batch_size`：16 → 8（甚至 4）。
或把 `max_length` 從 512 降到 256。

**Q：HuggingFace 下載很慢或失敗？**
如果 GPU 機器網路不佳，可先在有網路的機器下載模型：
```bash
pip install huggingface_hub
huggingface-cli download hfl/chinese-roberta-wwm-ext --local-dir ./roberta-model
```
然後在 `config.py` 中把 `pretrained_model` 改為本地路徑。

**Q：同時訓練 TextCNN 和 BERT？**
可以，兩個訓練互不干擾，只要 GPU 記憶體夠（BERT 約需 4-6GB）。
建議先跑 BERT，再跑 TextCNN。
