"""
資料讀取與前處理模組。

【白話說明】
訓練資料是 tab 分隔的純文字檔，每行一筆：
  標籤名稱 \\t 陳情文本
例如：
  臺東市公所\\t您好，我想反映...

這個模組做三件事：
  1. 讀檔 → 取出 (標籤, 文本) 的 list
  2. 斷詞 → 用 jieba 把中文句子切成詞的 list
  3. 編碼 → 把每個詞換成詞彙表中的數字 ID，不夠長補 0，太長截斷
  最終產出 PyTorch DataLoader，可直接送進模型訓練。
"""

import re
import codecs
from pathlib import Path
from typing import Optional

import jieba
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────
# 1. 讀檔
# ─────────────────────────────────────────────────────────────

def load_raw_data(filepath: Path) -> tuple[list[str], list[str]]:
    """
    讀取 tab 分隔的訓練檔案。

    Returns:
        texts:  陳情文本 list
        labels: 標籤名稱 list（字串，例如 '臺東市公所'）
    """
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
            labels.append(label.strip())
            texts.append(text.strip())
    return texts, labels


def build_label_map(labels_path: Path) -> tuple[dict[str, int], list[str]]:
    """
    從 labels.txt 讀取標籤對照表。
    labels.txt 每行一個標籤名稱，行號即為 class id（從 0 開始）。

    Returns:
        label2id: {'臺東市公所': 0, '交通及觀光發展處': 1, ...}
        id2label: ['臺東市公所', '交通及觀光發展處', ...]
    """
    id2label = codecs.open(labels_path, "r", encoding="utf-8").read().strip().split("\n")
    label2id = {name: idx for idx, name in enumerate(id2label)}
    return label2id, id2label


def extract_labels_from_data(train_file: Path, save_path: Path) -> tuple[dict[str, int], list[str]]:
    """
    從訓練資料自動抽取所有標籤，並存成 labels.txt。
    （第一次執行時使用，之後直接讀 labels.txt）
    """
    _, labels = load_raw_data(train_file)
    id2label = sorted(set(labels))          # 排序確保每次一致
    label2id = {name: idx for idx, name in enumerate(id2label)}
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with codecs.open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(id2label))
    print(f"[data_loader] 標籤已存至 {save_path}，共 {len(id2label)} 類：")
    for idx, name in enumerate(id2label):
        print(f"  {idx}: {name}")
    return label2id, id2label


# ─────────────────────────────────────────────────────────────
# 2. 斷詞
# ─────────────────────────────────────────────────────────────

_HAN_RE = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")


_X000D_RE = re.compile(r"_x000D_", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """
    用 jieba 斷詞，同時過濾標點符號。

    【白話說明】
    中文沒有空格分隔詞，所以要先用 jieba 把句子切開。
    例如「台東市公所」→ ['台東市', '公所']
    標點和空白直接丟掉，因為對分類沒幫助。
    _x000D_ 是 Excel 匯出的換行符編碼，先清掉再斷詞。
    """
    text = _X000D_RE.sub("", text)   # 清除 Excel 換行符
    words = []
    for block in _HAN_RE.split(text):
        if _HAN_RE.match(block):
            words.extend(jieba.lcut(block))
    return words


# ─────────────────────────────────────────────────────────────
# 3. 詞彙表 + 數字編碼
# ─────────────────────────────────────────────────────────────

def load_vocab(vocab_path: Path) -> tuple[dict[str, int], list[str]]:
    """
    讀取 vocab.txt（每行一個詞，行號即 ID）。

    Returns:
        word2id: {'<PAD>': 0, '反映': 1, ...}
        vocab:   ['<PAD>', '反映', ...]
    """
    vocab = codecs.open(vocab_path, "r", encoding="utf-8").read().strip().split("\n")
    word2id = {w: i for i, w in enumerate(vocab)}
    return word2id, vocab


def encode_texts(
    texts: list[str],
    word2id: dict[str, int],
    seq_length: int,
    cache_path: Path | None = None,
) -> np.ndarray:
    """
    把文本轉成數字 ID 陣列。

    【白話說明】
    神經網路只懂數字，不懂文字。
    所以把每個詞查詞彙表換成 ID 數字。
    - 文本太短 → 後面補 0（<PAD> 的 ID）
    - 文本太長 → 截斷到 seq_length
    最終每筆資料都是長度相同的整數陣列。

    cache_path 若指定，第一次編碼後存成 .npz 快取，
    下次直接載入，略過耗時的 jieba 斷詞。
    """
    # 有快取就直接讀
    if cache_path and cache_path.exists():
        print(f"[data_loader] 載入快取：{cache_path}")
        return np.load(cache_path)["x"]

    print(f"[data_loader] 開始斷詞編碼（{len(texts)} 筆）...")
    encoded = []
    for i, text in enumerate(texts):
        tokens = tokenize(text)
        ids = [word2id[w] for w in tokens if w in word2id]
        if len(ids) >= seq_length:
            ids = ids[:seq_length]
        else:
            ids = ids + [0] * (seq_length - len(ids))
        encoded.append(ids)
        if (i + 1) % 5000 == 0:
            print(f"  ... {i + 1}/{len(texts)}")

    result = np.array(encoded, dtype=np.int64)

    # 存快取
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, x=result)
        print(f"[data_loader] 快取已存：{cache_path}")

    return result


# ─────────────────────────────────────────────────────────────
# 4. PyTorch Dataset & DataLoader
# ─────────────────────────────────────────────────────────────

class CaseDataset(Dataset):
    """
    PyTorch Dataset：把 (X, y) 配對包裝起來。

    【白話說明】
    DataLoader 需要一個 Dataset 物件才能分批讀取資料。
    Dataset 只需要實作三個方法：
      __len__：告訴 DataLoader 總共有幾筆資料
      __getitem__：拿第 i 筆資料
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_dataloader(
    filepath: Path,
    word2id: dict[str, int],
    label2id: dict[str, int],
    seq_length: int,
    batch_size: int,
    shuffle: bool = True,
    cache_dir: Path | None = None,
) -> DataLoader:
    """
    一步完成：讀檔 → 斷詞 → 編碼 → DataLoader。
    cache_dir 若指定，編碼結果會快取，加速第二次以後的啟動。
    """
    texts, labels = load_raw_data(filepath)

    cache_path = (cache_dir / f"{filepath.stem}_encoded.npz") if cache_dir else None
    x = encode_texts(texts, word2id, seq_length, cache_path=cache_path)

    y = np.array([label2id[lb] for lb in labels], dtype=np.int64)
    dataset = CaseDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ─────────────────────────────────────────────────────────────
# 快速驗證（直接執行此檔時使用）
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import Config
    cfg = Config()

    # 建立 / 載入標籤
    if cfg.labels_path.exists():
        label2id, id2label = build_label_map(cfg.labels_path)
    else:
        label2id, id2label = extract_labels_from_data(cfg.train_file, cfg.labels_path)

    # 載入詞彙表
    word2id, vocab = load_vocab(cfg.vocab_file)
    print(f"\n詞彙表大小：{len(vocab)}")

    # 讀一筆資料看看
    texts, labels = load_raw_data(cfg.train_file)
    print(f"訓練筆數：{len(texts)}")
    print(f"範例文本：{texts[0][:50]}...")
    print(f"對應標籤：{labels[0]}")

    tokens = tokenize(texts[0])
    print(f"斷詞結果（前 10）：{tokens[:10]}")

    # 建立 DataLoader
    loader = build_dataloader(
        cfg.train_file, word2id, label2id, cfg.seq_length, batch_size=4
    )
    x_batch, y_batch = next(iter(loader))
    print(f"\n第一個 batch shape — X: {x_batch.shape}, y: {y_batch.shape}")
    print(f"標籤 ID：{y_batch.tolist()}")
    print(f"對應名稱：{[id2label[i] for i in y_batch.tolist()]}")
