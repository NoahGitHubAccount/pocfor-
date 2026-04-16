"""
BERT 訓練主程式。

【白話說明 — 和 TextCNN 訓練的差異】

TextCNN 訓練：
  自己寫 for loop → forward → loss → backward → optimizer.step()
  要手動處理 early stopping、scheduler、gradient clipping...

BERT 訓練：
  用 HuggingFace Trainer — 它把所有最佳實踐都包進去了：
  - 自動 warmup + cosine scheduler
  - 自動 gradient accumulation（小 batch 模擬大 batch）
  - 自動 mixed precision（FP16，省 GPU 記憶體、加速）
  - 自動 evaluation + best model saving
  - 自動 logging

  我們只需要「設定參數 + 丟資料進去」。

執行方式：
  docker compose run train
  或：python src/train.py
"""

import sys
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_loader import (
    build_dataloader,
    build_label_map,
    extract_labels_from_data,
    load_raw_data,
    CaseDataset,
)
from model import build_model


def compute_metrics(eval_pred):
    """Trainer 每次 evaluate 時呼叫，計算準確度。"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def train():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] 使用裝置：{device}")

    # ── 標籤 ──────────────────────────────────────────────
    if cfg.labels_path.exists():
        label2id, id2label = build_label_map(cfg.labels_path)
    else:
        label2id, id2label = extract_labels_from_data(cfg.train_file, cfg.labels_path)

    # ── Tokenizer（BERT 內建，不需要 jieba）──────────────
    print(f"[train] 載入 tokenizer：{cfg.pretrained_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model)

    # ── 資料集 ────────────────────────────────────────────
    print("[train] 準備訓練資料...")
    train_texts, train_labels = load_raw_data(cfg.train_file)
    train_label_ids = [label2id[lb] for lb in train_labels]
    train_dataset = CaseDataset(train_texts, train_label_ids, tokenizer, cfg.max_length)

    print("[train] 準備驗證資料...")
    val_texts, val_labels = load_raw_data(cfg.val_file)
    val_label_ids = [label2id[lb] for lb in val_labels]
    val_dataset = CaseDataset(val_texts, val_label_ids, tokenizer, cfg.max_length)

    # ── 模型 ──────────────────────────────────────────────
    model = build_model(cfg)

    # ── 訓練設定 ──────────────────────────────────────────
    output_dir = str(cfg.model_dir)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        fp16=(device == "cuda"),          # GPU 時自動用半精度加速
        dataloader_num_workers=0,
        report_to="none",                 # 不送 wandb 等外部服務
    )

    # ── Trainer ───────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"\n[train] 開始訓練，{cfg.epochs} epochs")
    print("-" * 60)
    trainer.train()

    # ── 儲存最佳模型 + tokenizer ─────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ── 最終評估 ──────────────────────────────────────────
    print("\n[train] 最終驗證集評估：")
    metrics = trainer.evaluate()
    print(f"  val_loss     = {metrics['eval_loss']:.4f}")
    print(f"  val_accuracy = {metrics['eval_accuracy']:.4f}")

    # 詳細分類報告
    preds = trainer.predict(val_dataset)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = [val_label_ids[i] for i in range(len(val_label_ids))]
    print("\n[train] 分類報告：")
    print(classification_report(y_true, y_pred, target_names=id2label))

    print(f"\n[train] 模型已存至：{output_dir}")


if __name__ == "__main__":
    train()
