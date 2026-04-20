# GPU 機器訓練操作手冊

> 本文件說明如何在 GPU 機器上執行訓練，並將成果帶回主機繼續開發。
> 執行者只需要 Docker 環境，不需要了解程式細節。

---

## 一、前置確認（在 GPU 機器上）

### 1.1 確認 NVIDIA 驅動

開啟 **命令提示字元 (cmd)** 或 **PowerShell**：

```powershell
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

### 1.2 安裝 Docker（二擇一）

先確認 Docker 是否已安裝：

```powershell
docker --version
docker compose version
```

若未安裝或 Docker Desktop 無法啟動，選擇以下其中一種方式：

---

#### 方法 A：Docker Desktop（能正常啟動者用此方式）

1. 前往 https://www.docker.com/products/docker-desktop/ 下載安裝檔
2. 執行安裝，勾選 **Use WSL 2 instead of Hyper-V**
3. 安裝完成後重新開機

Docker Desktop **內建支援 NVIDIA GPU**（透過 WSL2），不需要額外安裝 NVIDIA Container Toolkit。確認：

1. NVIDIA 驅動版本 ≥ 525.60
2. Docker Desktop → Settings → Resources → WSL integration → 啟用你的 WSL distro

驗證：

```powershell
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

---

#### 方法 B：WSL2 + Docker Engine（Docker Desktop 無法啟動時用此方式）

> **適用情境**：Docker Desktop 安裝後無法 starting，或出現 Hyper-V 相關錯誤。
> WSL2 Ubuntu 已存在者可直接從 Step 2 開始。

**Step 1：確認 WSL2 與 Ubuntu 已安裝**

以系統管理員開啟 PowerShell：

```powershell
wsl --set-default-version 2
wsl -l -v
```

確認有 Ubuntu 且 VERSION 為 2。若無 Ubuntu：

```powershell
wsl --install -d Ubuntu
```

**Step 2：進入 Ubuntu WSL2**

```powershell
wsl -d Ubuntu
```

**Step 3：在 Ubuntu 內安裝 Docker Engine**

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
docker --version
```

**Step 4：安裝 NVIDIA Container Toolkit**

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Step 5：驗證 GPU 可用**

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

看到 GPU 資訊代表成功。

> **注意**：使用方法 B 時，後續所有 docker 指令都需在 **WSL2 Ubuntu 終端機內**執行（不是 PowerShell）。
> Windows 路徑對應：`C:\smart-case\` → WSL2 內為 `/mnt/c/smart-case/`

---

## 二、搬移檔案到 GPU 機器

### 需要複製的內容

| 來源（你的筆電）| 說明 |
|----------------|------|
| `poc/` 整個資料夾 | 程式碼、Dockerfile |
| `2026打包專案包/安裝程式包/textCNN_主程式/newData_1000_title/` | 訓練資料 |

### 2.1 直接複製（GPU 機器是 Windows）

在 GPU 機器上建立工作目錄，將檔案複製進去：

```powershell
mkdir C:\smart-case\poc
mkdir C:\smart-case\data
```

用檔案總管或隨身碟，將以下內容複製到 GPU 機器：

| 來源 | 複製到 |
|------|--------|
| `poc/` 整個資料夾 | `C:\smart-case\poc\` |
| `newData_1000_title/` 內的所有檔案 | `C:\smart-case\data\` |

### 2.2 用 scp 複製（遠端情況）

```powershell
# 在你的筆電上執行
scp -r "D:/POC_for_智慧分案/poc" user@GPU機器IP:C:/smart-case/
scp -r "D:/POC_for_智慧分案/2026打包專案包/安裝程式包/textCNN_主程式/newData_1000_title" user@GPU機器IP:C:/smart-case/data/
```

目標目錄結構（GPU 機器上）：

```
C:\smart-case\
├── poc\
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── src\
└── data\
    ├── cnews.train.txt
    ├── cnews.val.txt
    ├── cnews.test.txt
    ├── vocab.txt
    └── vector_word.npz
```

---

## 三、修改 Dockerfile 以啟用 GPU

在 GPU 機器上，編輯 `poc/Dockerfile`，將 PyTorch 從 CPU 版改為 GPU 版：
  
```powershell
cd C:\smart-case\poc
```

用文字編輯器（記事本或 VS Code）開啟 `Dockerfile`，找到這行：

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

**方法 A（PowerShell）：**
```powershell
cd C:\smart-case\poc
docker build -t smart-case-classifier:gpu .
```

**方法 B（WSL2 Ubuntu）：**
```bash
cd /mnt/c/smart-case/poc
docker build -t smart-case-classifier:gpu .
```

等待完成（第一次需下載 PyTorch GPU 版，約 500MB-1GB）。

---

## 五、執行訓練

**方法 A（Docker Desktop / PowerShell）：**

```powershell
docker run --rm --gpus all `
  -e PYTHONUNBUFFERED=1 `
  -v C:\smart-case\data:/app/data:ro `
  -v C:\smart-case\poc\checkpoints:/app/checkpoints `
  -v C:\smart-case\poc\src:/app/src `
  smart-case-classifier:gpu `
  python src/train.py 2>&1 | Tee-Object -FilePath C:\smart-case\train_log.txt
```

**方法 B（WSL2 Ubuntu 終端機）：**

```bash
docker run --rm --gpus all \
  -e PYTHONUNBUFFERED=1 \
  -v /mnt/c/smart-case/data:/app/data:ro \
  -v /mnt/c/smart-case/poc/checkpoints:/app/checkpoints \
  -v /mnt/c/smart-case/poc/src:/app/src \
  smart-case-classifier:gpu \
  python src/train.py 2>&1 | tee /mnt/c/smart-case/train_log.txt
```

> `Tee-Object` / `tee` 會同時顯示在螢幕上並存到 `train_log.txt`，方便之後回報。

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

```powershell
dir C:\smart-case\poc\checkpoints\
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

| 檔案                          | 說明                           |
| --------------------------- | ---------------------------- |
| `checkpoints/best_model.pt` | **必要**，訓練好的模型權重              |
| `checkpoints/labels.txt`    | **必要**，標籤對照表                 |
| `train_log.txt`             | **必要**，訓練紀錄（用來回報給 Claude 確認） |

### 複製回筆電

如果 GPU 機器就在手邊，直接用隨身碟或網路芳鄰複製以下檔案回筆電：

| GPU 機器上的路徑                                    | 複製到筆電                              |
| --------------------------------------------- | ---------------------------------- |
| `C:\smart-case\poc\checkpoints\best_model.pt` | `D:\POC_for_智慧分案\poc\checkpoints\` |
| `C:\smart-case\poc\checkpoints\labels.txt`    | `D:\POC_for_智慧分案\poc\checkpoints\` |
| `C:\smart-case\train_log.txt`                 | `D:\POC_for_智慧分案\`                 |

或用 scp（需 GPU 機器開啟 SSH）：

```powershell
scp user@GPU機器IP:C:/smart-case/poc/checkpoints/best_model.pt D:\POC_for_智慧分案\poc\checkpoints\
scp user@GPU機器IP:C:/smart-case/poc/checkpoints/labels.txt D:\POC_for_智慧分案\poc\checkpoints\
scp user@GPU機器IP:C:/smart-case/train_log.txt D:\POC_for_智慧分案\
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
