"""
BERT 資料載入模組。

【白話說明 — 和 TextCNN 版的差異】
TextCNN 版：jieba 斷詞 → word2vec 查表 → ID 序列
BERT 版：  BERT Tokenizer → subword tokens → ID 序列

BERT Tokenizer 會自動處理中文，不需要 jieba。
它使用「子詞」（subword）切分，例如：
  「環境保護局」→ ['環', '境', '保', '護', '局']
  不認識的詞會被拆成更小的片段，不會出現 OOV（詞彙表外）問題。

Tokenizer 還會自動加上特殊 token：
  [CLS] 文本內容 [SEP]
  其中 [CLS] 的輸出向量用來做分類。
"""

import re
import codecs
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from config import Config

_X000D_RE = re.compile(r"_x000D_", re.IGNORECASE)


def load_raw_data(filepath: Path) -> tuple[list[str], list[str]]:
    """讀取 tab 分隔的資料檔，回傳 (texts, labels)。"""
    texts, labels = [], []
    with codecs.open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                continue
            label, text = parts
            text = _X000D_RE.sub("", text)   # 清除 Excel 換行符
            labels.append(label.strip())
            texts.append(text.strip())
    return texts, labels


def build_label_map(labels_path: Path) -> tuple[dict[str, int], list[str]]:
    id2label = codecs.open(labels_path, "r", encoding="utf-8").read().strip().split("\n")
    label2id = {name: idx for idx, name in enumerate(id2label)}
    return label2id, id2label


def extract_labels_from_data(train_file: Path, save_path: Path) -> tuple[dict[str, int], list[str]]:
    _, labels = load_raw_data(train_file)
    id2label = sorted(set(labels))
    label2id = {name: idx for idx, name in enumerate(id2label)}
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with codecs.open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(id2label))
    print(f"[data_loader] 標籤已存至 {save_path}，共 {len(id2label)} 類")
    return label2id, id2label


class CaseDataset(Dataset):
    """
    BERT 版 Dataset。

    【白話說明】
    和 TextCNN 版的差異：
    - TextCNN 只需要 input_ids（詞 ID）
    - BERT 額外需要 attention_mask（標記哪些位置是真的文字、哪些是補零）
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_length: int,
    ):
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


def build_dataloader(
    filepath: Path,
    tokenizer: AutoTokenizer,
    label2id: dict[str, int],
    max_length: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """一步完成：讀檔 → Tokenize → DataLoader。"""
    texts, labels = load_raw_data(filepath)
    label_ids = [label2id[lb] for lb in labels]
    print(f"[data_loader] Tokenizing {len(texts)} 筆（{filepath.name}）...")
    dataset = CaseDataset(texts, label_ids, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
