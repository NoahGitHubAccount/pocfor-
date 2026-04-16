"""
詞向量載入模組。

【白話說明】
神經網路的第一層是「Embedding Layer」（嵌入層）。
它的本質是一張查找表：輸入詞 ID → 輸出一個 100 維的數字向量。

這個向量是用 word2vec 預先訓練好的，
意義相近的詞（如「道路」和「馬路」）的向量在空間中距離很近。

這份檔案做一件事：
  把舊系統訓練好的 vector_word.npz 載入，
  轉成 PyTorch 的 Embedding Layer，供模型直接使用。
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def load_embedding_matrix(vector_npz: Path) -> np.ndarray:
    """
    讀取 .npz 檔，回傳 numpy 矩陣。
    shape: (vocab_size, embedding_dim) = (8000, 100)
    """
    with np.load(vector_npz) as data:
        embeddings = data["embeddings"].astype(np.float32)
    print(f"[word2vec] 詞向量載入完成：shape={embeddings.shape}, dtype={embeddings.dtype}")
    return embeddings


def build_embedding_layer(vector_npz: Path, freeze: bool = False) -> nn.Embedding:
    """
    從 .npz 建立 PyTorch Embedding Layer。

    Args:
        vector_npz: vector_word.npz 路徑
        freeze:     True = 訓練時不更新詞向量（保留 word2vec 原始值）
                    False = 允許微調（通常效果更好）

    【白話說明】
    Embedding Layer 就是那張「詞 ID → 向量」的查找表。
    from_pretrained() 把 word2vec 的結果直接填進這張表，
    不用從隨機值重頭訓練，收斂更快、效果更好。

    freeze=False 表示訓練時讓模型微調這些向量，
    讓它們更符合「分案」這個特定任務的語意。
    """
    embeddings = load_embedding_matrix(vector_npz)
    weight = torch.tensor(embeddings, dtype=torch.float32)
    embedding_layer = nn.Embedding.from_pretrained(weight, freeze=freeze, padding_idx=0)
    print(f"[word2vec] Embedding Layer 建立完成：freeze={freeze}")
    return embedding_layer


# ─── 快速驗證 ────────────────────────────────────────────────
if __name__ == "__main__":
    from config import Config
    cfg = Config()

    emb_layer = build_embedding_layer(cfg.vector_npz, freeze=False)

    # 用幾個詞 ID 測試查找
    test_ids = torch.tensor([[1, 2, 3, 0]])   # 0 是 PAD
    output = emb_layer(test_ids)
    print(f"輸入 shape：{test_ids.shape}")
    print(f"輸出 shape：{output.shape}")   # 應為 (1, 4, 100)
    print(f"PAD 向量（應全零）：{output[0][3][:5]}")
