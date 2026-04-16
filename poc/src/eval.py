"""
測試集評估腳本 — 等 checkpoint 回來後執行。

產出：
  1. 測試集準確度
  2. 分類報告（每個局處的 precision / recall / f1）
  3. 混淆矩陣
"""

import sys
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_loader import build_dataloader, build_label_map, load_vocab
from model import TextCNN


def evaluate():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入標籤 + 詞彙表
    label2id, id2label = build_label_map(cfg.labels_path)
    word2id, _ = load_vocab(cfg.vocab_file)

    # 載入測試集
    cache_dir = cfg.model_path.parent / "cache"
    test_loader = build_dataloader(
        cfg.test_file, word2id, label2id, cfg.seq_length,
        cfg.batch_size, shuffle=False, cache_dir=cache_dir
    )

    # 載入模型
    model = TextCNN(cfg).to(device)
    model.load_state_dict(torch.load(cfg.model_path, map_location=device))
    model.eval()
    print(f"[eval] 模型載入：{cfg.model_path}")

    # 推論
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    # 報告
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n測試集準確度：{acc:.4f}（{sum(p == l for p, l in zip(all_preds, all_labels))}/{len(all_labels)}）")

    print("\n分類報告：")
    print(classification_report(all_labels, all_preds, target_names=id2label))

    print("混淆矩陣：")
    cm = confusion_matrix(all_labels, all_preds)
    # 表頭
    header = "            " + "  ".join(f"{name[:4]:>4}" for name in id2label)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>4}" for v in row)
        print(f"{id2label[i][:6]:<8}  {row_str}")


if __name__ == "__main__":
    evaluate()
