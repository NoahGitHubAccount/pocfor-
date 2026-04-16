"""
訓練主程式。

【白話說明】
訓練 = 反覆讓模型看資料，每次看完就調整參數，讓預測越來越準。

流程：
  for 每個 epoch（完整看過一次訓練集）:
      for 每個 batch（一小批資料）:
          1. 模型預測 → 算 loss（預測有多錯）
          2. Backpropagation（反向傳播）→ 算每個參數的梯度
          3. Optimizer 根據梯度更新參數
      用驗證集評估，若準確度提升 → 存下模型
      若連續 N 個 epoch 沒進步 → early stopping（提早結束）

執行方式：
  docker compose run train
  或直接：python src/train.py
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 確保 /app/src 在 import 路徑
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_loader import build_dataloader, build_label_map, extract_labels_from_data, load_vocab
from model import TextCNN


def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> tuple[float, float]:
    """
    在 val/test 集上計算 loss 和 accuracy。
    model.eval() 會關掉 Dropout，確保推論一致。
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    return total_loss / total, correct / total


def train():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] 使用裝置：{device}")

    # ── 標籤 ──────────────────────────────────────────────────
    if cfg.labels_path.exists():
        label2id, id2label = build_label_map(cfg.labels_path)
    else:
        label2id, id2label = extract_labels_from_data(cfg.train_file, cfg.labels_path)

    # ── 詞彙表 ────────────────────────────────────────────────
    word2id, _ = load_vocab(cfg.vocab_file)

    # ── DataLoader（快取斷詞結果，加速第二次以後啟動）──────────
    cache_dir = cfg.model_path.parent / "cache"
    print("[train] 載入資料中（首次執行需斷詞，約 3-5 分鐘）...")
    train_loader = build_dataloader(cfg.train_file, word2id, label2id, cfg.seq_length, cfg.batch_size, shuffle=True,  cache_dir=cache_dir)
    val_loader   = build_dataloader(cfg.val_file,   word2id, label2id, cfg.seq_length, cfg.batch_size, shuffle=False, cache_dir=cache_dir)
    print(f"[train] 訓練批次：{len(train_loader)}，驗證批次：{len(val_loader)}")

    # ── 模型、損失函數、優化器 ────────────────────────────────
    model = TextCNN(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    # 驗證集 loss 連續 3 次沒下降就降低學習率
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    # ── 訓練迴圈 ──────────────────────────────────────────────
    best_val_acc = 0.0
    no_improve   = 0
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[train] 開始訓練，共 {cfg.epochs} epochs，early stop patience={cfg.early_stop_patience}")
    print("-" * 60)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            # 梯度裁剪：防止梯度爆炸（與舊系統 clip=6.0 一致）
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=6.0)
            optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)

        train_loss = total_loss / total
        train_acc  = correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        # 儲存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), cfg.model_path)
            print(f"  ✓ 最佳模型已儲存（val_acc={val_acc:.4f}）")
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                print(f"\n[train] Early stopping：連續 {cfg.early_stop_patience} epochs 驗證集無改善")
                break

    print(f"\n[train] 訓練完成。最佳驗證準確度：{best_val_acc:.4f}")
    print(f"[train] 模型已存至：{cfg.model_path}")


if __name__ == "__main__":
    train()
