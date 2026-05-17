"""
BERT 版測試集評估腳本 — 等 checkpoint 回來後執行。
"""

import sys
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_loader import build_label_map, load_raw_data, CaseDataset
from torch.utils.data import DataLoader


def evaluate():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label2id, id2label = build_label_map(cfg.labels_path)

    model_dir = str(cfg.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    print(f"[eval] BERT 模型載入：{model_dir}")

    texts, labels = load_raw_data(cfg.test_file)
    label_ids = [label2id[lb] for lb in labels]
    dataset = CaseDataset(texts, label_ids, tokenizer, cfg.max_length)
    loader = DataLoader(dataset, batch_size=cfg.batch_size * 2, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n測試集準確度：{acc:.4f}（{sum(p == l for p, l in zip(all_preds, all_labels))}/{len(all_labels)}）")
    print("\n分類報告：")
    print(classification_report(all_labels, all_preds, target_names=id2label))


if __name__ == "__main__":
    evaluate()
