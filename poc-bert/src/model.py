"""
BERT 文本分類模型。

【白話說明 — 和 TextCNN 的根本差異】

TextCNN：
  「我自己從零學中文」→ 用 word2vec 學詞義，再用 CNN 學分類
  需要大量訓練資料才能學好

BERT：
  「我已經讀過整個中文維基百科」→ 已經懂中文語意
  微調 = 教一個已經懂中文的人「怎麼分案」
  需要的訓練資料少很多，效果好很多

技術上：
  BERT 是一個 Transformer 編碼器，有 12 層 attention layer。
  我們在最上面加一層 Linear（全連接層）做分類。
  微調時，整個 BERT + 分類頭一起更新參數。
"""

import sys
from pathlib import Path

import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

sys.path.insert(0, str(Path(__file__).parent))
from config import Config


def build_model(config: Config) -> nn.Module:
    """
    建立 BERT 文本分類模型。

    HuggingFace 的 AutoModelForSequenceClassification 會自動：
      1. 載入預訓練 BERT 權重
      2. 在最上面加一層 Linear(768 → num_classes)
      3. 內建 loss 計算（CrossEntropy）

    不需要自己寫 nn.Module，HuggingFace 都包好了。
    """
    model_config = AutoConfig.from_pretrained(
        config.pretrained_model,
        num_labels=config.num_classes,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.pretrained_model,
        config=model_config,
    )
    print(f"[model] BERT 模型載入完成：{config.pretrained_model}")
    print(f"[model] 參數量：{sum(p.numel() for p in model.parameters()):,}")
    return model
