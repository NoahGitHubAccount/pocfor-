# GPU 機器訓練操作手冊

> 本文件說明如何在 GPU 機器上執行訓練，並將成果帶回主機繼續開發。
> 執行者只需要 Docker 環境，不需要了解程式細節。

---

## 一、前置確認（在 GPU 機器上）

### 1.1 確認 NVIDIA 驅動

```bash
nvidia-smi
```

預期看到：

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI ...   Driver Version: ...   CUDA Version: xx.x |
+----------------------------------------------------------------------+
| GPU  0  ...  (顯示你的 GPU 型號)
```

若指令不存在 → 需先安裝 NVIDIA 驅動，停止後續步驟。

---

### 1.2 確認 Docker 已安裝

```bash
docker --version
docker compose version
```

若未安裝：

```bash
# Ubuntu / Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

---

### 1.3 確認 NVIDIA Container Toolkit（讓 Docker 能用 GPU）

```bash
nvidia-container-cli --version
```

若未安裝：

```bash
# Ubuntu
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

驗證：

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

看到 GPU 資訊代表成功。

---

## 二、搬移檔案到 GPU 機器

### 需要複製的內容

| 來源（你的筆電）| 說明 |
|----------------|------|
| `poc/` 整個資料夾 | 程式碼、Dockerfile |
| `2026打包專案包/安裝程式包/textCNN_主程式/newData_1000_title/` | 訓練資料 |

### 2.1 用 scp 複製（在你的筆電上執行）

```bash
# 複製程式碼
scp -r "D:/POC_for_智慧分案/poc" user@GPU機器IP:/home/user/smart-case/

# 複製訓練資料
scp -r "D:/POC_for_智慧分案/2026打包專案包/安裝程式包/textCNN_主程式/newData_1000_title" \
    user@GPU機器IP:/home/user/smart-case/data/
```

### 2.2 用隨身碟複製（離線情況）

複製以下兩個資料夾到隨身碟，再插到 GPU 機器：
- `poc/`
- `newData_1000_title/`

目標目錄結構（GPU 機器上）：

```
/home/user/smart-case/
├── poc/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── src/
└── data/
    ├── cnews.train.txt
    ├── cnews.val.txt
    ├── cnews.test.txt
    ├── vocab.txt
    └── vector_word.npz
```

---

## 三、修改 Dockerfile 以啟用 GPU

在 GPU 機器上，編輯 `poc/Dockerfile`，將 PyTorch 從 CPU 版改為 GPU 版：

```bash
cd /home/user/smart-case/poc
```

找到這行：

```dockerfile
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

改為（CUDA 11.8，請依你的 CUDA 版本選擇）：

```dockerfile
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu118
```

> 常見 CUDA 版本對應：
> - CUDA 11.8 → `cu118`
> - CUDA 12.1 → `cu121`
> - CUDA 12.4 → `cu124`
> 用 `nvidia-smi` 確認你的 CUDA Version，選最接近的。

---

## 四、Build Image

```bash
cd /home/user/smart-case/poc
docker build -t smart-case-classifier:gpu .
```

等待完成（第一次需下載 PyTorch GPU 版，約 500MB-1GB）。

---

## 五、執行訓練

```bash
docker run --rm --gpus all \
  -e PYTHONUNBUFFERED=1 \
  -v /home/user/smart-case/data:/app/data:ro \
  -v /home/user/smart-case/poc/checkpoints:/app/checkpoints \
  -v /home/user/smart-case/poc/src:/app/src \
  smart-case-classifier:gpu \
  python src/train.py 2>&1 | tee /home/user/smart-case/train_log.txt
```

> `tee` 會同時顯示在螢幕上，並存到 `train_log.txt`，方便之後回報。

### 預期輸出（正常狀況）

```
[train] 載入資料中（首次執行需斷詞，約 3-5 分鐘）...
[data_loader] 開始斷詞編碼（23329 筆）...
  ... 5000/23329
  ... 10000/23329
  ...
[train] 訓練批次：364，驗證批次：45

開始訓練，共 30 epochs，early stop patience=5
------------------------------------------------------------
Epoch 001/030 | train loss=x.xxxx acc=x.xxxx | val loss=x.xxxx acc=x.xxxx | xx.xs
  ✓ 最佳模型已儲存（val_acc=x.xxxx）
Epoch 002/030 | ...
...
[train] 訓練完成。最佳驗證準確度：x.xxxx
[train] 模型已存至：/app/checkpoints/best_model.pt
```

### 如果出現 GPU 相關錯誤

確認 train.py 中的 device 設定，在 GPU 機器上需改為：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

目前 `train.py` 寫死 `cpu`，需修改這一行後重新執行。

---

## 六、確認訓練成果

訓練完成後，確認以下檔案存在：

```bash
ls -lh /home/user/smart-case/poc/checkpoints/
```

預期看到：

```
best_model.pt       （訓練好的模型，應有數 MB）
labels.txt          （9 個標籤名稱）
cache/              （斷詞快取，可不帶回）
```

---

## 七、把成果帶回筆電

### 需要複製回來的檔案

| 檔案 | 說明 |
|------|------|
| `checkpoints/best_model.pt` | **必要**，訓練好的模型權重 |
| `checkpoints/labels.txt` | **必要**，標籤對照表 |
| `train_log.txt` | **必要**，訓練紀錄（用來回報給 Claude 確認） |

### 用 scp 複製回筆電（在筆電上執行）

```bash
# 在 WSL Rocky9 中執行
scp user@GPU機器IP:/home/user/smart-case/poc/checkpoints/best_model.pt \
    /mnt/d/POC_for_智慧分案/poc/checkpoints/

scp user@GPU機器IP:/home/user/smart-case/poc/checkpoints/labels.txt \
    /mnt/d/POC_for_智慧分案/poc/checkpoints/

scp user@GPU機器IP:/home/user/smart-case/train_log.txt \
    /mnt/d/POC_for_智慧分案/
```

---

## 八、回報給 Claude 確認

完成後，請在 Claude Code 對話中提供以下內容：

### 8.1 訓練紀錄（必要）

把 `train_log.txt` 的最後 30 行貼上，或直接說：

> 「訓練完成，最佳驗證準確度 x.xx，共跑了 N 個 epoch」

### 8.2 確認檔案存在

在 Claude Code 的終端執行：

```
ls D:/POC_for_智慧分案/poc/checkpoints/
```

確認 `best_model.pt` 和 `labels.txt` 都在。

### 8.3 Claude 會進行的驗證

收到回報後，Claude 會：
1. 確認 checkpoint 檔案大小合理（應大於 1MB）
2. 用測試集評估準確度
3. 用範例文字跑預測，確認輸出格式正確
4. 更新 PLAN.md 狀態，進入下一步

---

## 九、常見問題

**Q：`nvidia-smi` 有看到 GPU，但 Docker 訓練時沒用到 GPU？**
確認執行指令有加 `--gpus all`，並確認 NVIDIA Container Toolkit 已安裝並 restart docker。

**Q：`train.py` 沒有輸出任何文字？**
確認有加 `-e PYTHONUNBUFFERED=1`，否則 Python 輸出會被緩衝。

**Q：斷詞快取 (cache/) 要不要帶回？**
不必要。帶回後放在 `poc/checkpoints/cache/` 可加速未來重新訓練的啟動時間，但不影響功能。

**Q：GPU 記憶體不夠（OOM）？**
在 `config.py` 中調小 `batch_size`：從 64 改為 32 或 16。
