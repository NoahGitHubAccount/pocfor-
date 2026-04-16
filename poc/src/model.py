"""
TextCNN 模型定義（PyTorch 重寫）。

【白話說明】
原本的系統用 TensorFlow 1.x 寫的 CNN，這裡用 PyTorch 重寫相同架構。

CNN 用在文字分類的直覺：
  把一句話想像成一張「窄長的圖片」（長度 × 詞向量維度）。
  用不同寬度的「濾鏡」（filter_sizes = [2, 3, 4]）掃過這張圖，
  每個濾鏡負責抓「2個詞、3個詞、4個詞組合」的特徵。
  最後把所有特徵壓縮成一個向量，送進全連接層分類。

架構圖：
  輸入 (batch, 600)
    → Embedding (batch, 600, 100)
    → 3組卷積+池化 (每組 128 個濾鏡，大小 2/3/4)
    → 拼接 (batch, 128×3=384)
    → Dropout (防止過擬合)
    → 全連接 (384 → 9)
    → Softmax → 各類別機率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from config import Config
from word2vec_helper import build_embedding_layer


class TextCNN(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # ── 嵌入層（詞向量查找表）──────────────────────────────
        self.embedding = build_embedding_layer(config.vector_npz, freeze=False)

        # ── 卷積層（多尺度，各抓不同長度的 n-gram 特徵）─────────
        # 每個 Conv1d：輸入通道=embedding_dim，輸出通道=num_filters，核大小=filter_size
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=config.embedding_dim,
                out_channels=config.num_filters,
                kernel_size=fs,
            )
            for fs in config.filter_sizes
        ])

        # ── Dropout（訓練時隨機關掉部分神經元，防止過擬合）────────
        self.dropout = nn.Dropout(config.dropout)

        # ── 全連接層（最終分類）──────────────────────────────────
        # 輸入維度 = num_filters × len(filter_sizes) = 128 × 3 = 384
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_length) — 詞 ID 序列
        Returns:
            logits: (batch_size, num_classes) — 各類別的原始分數（未經 softmax）
        """
        # Embedding: (batch, seq_len) → (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # Conv1d 需要 (batch, channels, length)，所以轉置後兩維
        # (batch, seq_len, embed_dim) → (batch, embed_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)

        # 卷積 + ReLU + 全局最大池化
        # 每個卷積核掃完整個句子，取最大值（最顯著的特徵）
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(embedded))          # (batch, num_filters, seq_len - fs + 1)
            c = F.max_pool1d(c, c.size(2))      # (batch, num_filters, 1)
            c = c.squeeze(2)                    # (batch, num_filters)
            pooled.append(c)

        # 把三種濾鏡的結果拼接起來
        # (batch, num_filters * len(filter_sizes)) = (batch, 384)
        cat = torch.cat(pooled, dim=1)

        # Dropout + 全連接
        out = self.dropout(cat)
        logits = self.fc(out)                   # (batch, num_classes)
        return logits


# ─── 快速驗證 ────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = Config()
    model = TextCNN(cfg)

    # 印出模型結構
    print(model)
    print()

    # 統計參數量
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數量：{total:,}")
    print(f"可訓練參數：{trainable:,}")

    # 跑一筆假資料確認 shape
    dummy_input = torch.randint(0, cfg.vocab_size, (4, cfg.seq_length))  # batch=4
    logits = model(dummy_input)
    print(f"\n輸入 shape：{dummy_input.shape}")
    print(f"輸出 logits shape：{logits.shape}")   # 應為 (4, 9)

    probs = torch.softmax(logits, dim=1)
    print(f"第一筆機率總和：{probs[0].sum().item():.4f}")  # 應為 1.0
