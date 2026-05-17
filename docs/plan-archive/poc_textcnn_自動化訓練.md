# 自動化訓練文件

## 版本紀錄

| 版本 | 發布日期   | 更新內容 |
|------|------------|----------|
| 1.0  | 2021.11.14 | 初版     |

---

## 文件說明

本自動化訓練提供系統自動從資料庫取得文本資料和所有類別資料，並將所有文本資料匯入訓練程式中進行訓練。

---

## 服務使用說明

- 連接資料庫取得類別和文本資料
- 類別自動寫入 `names.txt`
- 文本資料自動按照比例寫入 `train.txt`、`val.txt`、`test.txt`
- 讀取 `obj.txt` 取得訓練參數

---

## train：訓練文本分類

| 項目 | 說明 |
|------|------|
| 代碼 | `new_textCNN/text_train.py` |
| 說明 | 自動訓練文本資料 |

---

### 類別檔案（names.txt）

自動讀取資料庫中的類別資訊，寫入 `names.txt`，由上至下，編號從 0 開始。

**names.txt 範例內容（圖片截取）：**

```
臺東市公所
交通及觀光發展處
社會處
建設處
國際發展及計畫處
教育處
農業處
環境保護局
警察局
主計處
公共服務科
```

> 共 11 個分類單位，對應模型的 class 數量。

---

### 訓練參數檔案（obj.txt）

手動更改參數，各參數說明如下：

| 參數名稱                   | 說明                    |
| ---------------------- | --------------------- |
| `class`                | 類別數量                  |
| `epochs`               | 訓練迭代次數                |
| `batch`                | 一次訓練的樣本數量（依電腦記憶體大小調整） |
| `train_filename`       | 訓練檔路徑                 |
| `test_filename`        | 測試檔路徑                 |
| `val_filename`         | 驗證檔路徑                 |
| `vocab_filename`       | 訓練時自動儲存 vocab.txt 的路徑 |
| `vector_word_filename` | 訓練時自動儲存詞向量的路徑         |
| `vector_word_npz`      | 訓練時詞向量結果自動存放路徑        |

**obj.txt 實際範例內容（圖片截取）：**

```
class=11
epochs=100
batch=64
train_filename=./newDataAu_1000_title/cnews.train.txt
test_filename=./newDataAu_1000_title/cnews.test.txt
val_filename=./newDataAu_1000_title/cnews.val.txt
vocab_filename=./newDataAu_1000_title/vocab.txt
vector_word_filename=./newDataAu_1000_title/vector_word.txt
vector_word_npz=./newDataAu_1000_title/vector_word.npz
```
