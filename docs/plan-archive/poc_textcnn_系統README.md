# 智慧分案系統 README

---

## 1. 環境需求

| 套件            | 版本    | 用途           |
|-----------------|---------|----------------|
| Python          | 3.6.8   | 執行環境       |
| tensorflow      | 1.14.0  | 深度學習框架   |
| gensim          | 3.8.3   | 詞向量訓練     |
| jieba           | 0.39    | 中文斷詞       |
| scipy           | 1.4.1   | 科學計算       |
| numpy           | 1.19.4  | 數值運算       |
| scikit-learn    | 0.23.2  | 機器學習工具   |
| flask           | 1.1.2   | Web API 框架   |
| Flask-Cors      | 3.0.10  | 跨域請求支援   |

> ⚠️ 注意：此為舊版環境，tensorflow 1.14.0 僅支援 Python 3.6，不相容 Python 3.7+

---

## 2. CNN 卷積神經網路

### 模型配置參數（text_model.py 中的 TextConfig 類別）

```python
class TextConfig():
    embedding_size = 100      # 詞向量維度（word embedding dimension）
    vocab_size = 8000         # 詞彙表大小
    pre_trianing = None       # 是否使用預訓練詞向量（word2vec）

    seq_length = 600          # 句子最大長度（max length of sentence）
    num_classes = 10          # 分類數量（number of labels）

    num_filters = 128         # 卷積核數量（number of convolution kernel）
    filter_sizes = [2, 3, 4]  # 卷積核大小（size of convolution kernel）

    keep_prob = 0.5           # Dropout 保留率
    lr = 1e-3                 # 學習率（learning rate）
    lr_decay = 0.9            # 學習率衰減
    clip = 6.0                # 梯度裁剪閾值（gradient clipping threshold）
    l2_reg_lambda = 0.01      # L2 正則化參數

    num_epochs = 10           # 訓練迭代次數
    batch_size = 64           # 批次大小
    print_per_batch = 100     # 每隔幾個 batch 輸出結果

    train_filename = './data/cnews.train.txt'         # 訓練資料
    test_filename = './data/cnews.test.txt'           # 測試資料
    val_filename = './data/cnews.val.txt'             # 驗證資料
    vocab_filename = './data/vocab.txt'               # 詞彙表
    vector_word_filename = './data/vector_word.txt'   # 詞向量（word2vec 訓練結果）
    vector_word_npz = './data/vector_word.npz'        # 詞向量 numpy 格式
```

### 模型 CNN 結構說明

```
輸入層
  └── 詞向量矩陣（600 × 100）
        600 = 句子長度（最多 600 個詞）
        100 = 每個詞的 embedding 維度

卷積層
  └── 使用不同大小的卷積核（filter_sizes = [2, 3, 4]）進行卷積
        產生多組特徵圖（feature maps）

池化層
  └── max_pool 方式池化
        每個卷積核取最大值，壓縮特徵

全連接層 + Dropout
  └── 將池化結果全連接
        dropout 防止過擬合（keep_prob = 0.5）

輸出層
  └── Softmax 輸出各類別的機率
        輸出類別標籤
```

> 圖示說明：輸入為 600×100 的詞向量矩陣，透過多尺度卷積核擷取不同長度的 n-gram 特徵，再經 max pooling 壓縮，最後全連接 + softmax 輸出分類結果。

---

## 3. 訓練前數據預處理

主要訓練需要進行分詞處理，進行分詞訓練詞向量，模型將採用詞向量的形式進行輸入。

流程：
1. 原始文本 → jieba 中文斷詞
2. 斷詞結果 → gensim 訓練 word2vec 詞向量
3. 詞向量 → 作為 CNN 模型輸入

---

## 5. 預測程式運行步驟

在命令提示字元中執行：

```bash
python text_predict.py <要預測字串> <要預測筆數>
```

**預測結果範例（命令列輸出）：**

```
the 1-predict label: 臺東市公所     Probability: 0.40
the 2-predict label: 建設處         Probability: 0.28
the 3-predict label: 農業處         Probability: 0.11
the 4-predict label: 國際發展及計畫處 Probability: 0.10
the 5-predict label: 交通及觀光發展處 Probability: 0.07
```

> 注意：執行時會出現 TensorFlow 舊版 API 警告（`tf.reset_default_graph is deprecated`），屬正常現象，不影響執行結果。

---

## 6. 工具清單

| 工具      | 用途                       |
|-----------|----------------------------|
| WinSCP    | 上傳目錄檔案到 VM          |
| Putty     | Command line 遠端連線工具  |
| Postman   | 製作與測試 Web API         |

---

## 7. 安裝步驟（Linux / CentOS）

```bash
# 更新系統
yum update -y

# 安裝 Python 3.6.8
# 安裝包：https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tgz
yum install -y python3

# 安裝各套件
pip install tensorflow==1.14.0
pip install gensim==3.8.3
pip install jieba==0.39
pip install scipy==1.4.1
pip install numpy==1.19.4
pip install scikit-learn==0.23.2
pip install flask==1.1.2
pip install Flask-Cors==3.0.10
```
