"""
預測模組。

【白話說明】
訓練完的模型存在 checkpoints/best_model.pt。
這個模組負責「推論」：
  輸入一段陳情文字 → 輸出 Top-N 個可能的局處及其機率。

使用 Predictor 類別而非函式，原因：
  模型載入很慢（要讀檔、建立神經網路），
  但每次預測本身很快。
  用類別可以「只載入一次，預測多次」，
  適合 API 服務的使用情境。
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_loader import build_label_map, load_vocab, encode_texts
from model import TextCNN


class Predictor:
    """
    載入訓練好的模型，提供文本分類預測。
    初始化時載入模型（慢），predict() 可快速重複呼叫。
    """

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 載入詞彙表與標籤
        self.word2id, _ = load_vocab(self.cfg.vocab_file)
        _, self.id2label = build_label_map(self.cfg.labels_path)

        # 載入模型
        self.model = TextCNN(self.cfg).to(self.device)
        self.model.load_state_dict(
            torch.load(self.cfg.model_path, map_location=self.device)
        )
        self.model.eval()   # 關掉 Dropout，固定推論模式
        print(f"[predict] 模型載入完成：{self.cfg.model_path}")

    def predict(self, text: str, top_n: int = 3) -> list[dict]:
        """
        預測文本分類，回傳 Top-N 結果。

        Args:
            text:  陳情文字
            top_n: 回傳幾個候選（預設 3）

        Returns:
            [
              {"id": 6, "ou": "臺東市公所",  "probability": "0.5123"},
              {"id": 7, "ou": "警察局",       "probability": "0.2034"},
              ...
            ]
        """
        # 編碼：文字 → 詞 ID 序列
        x = encode_texts([text], self.word2id, self.cfg.seq_length)
        x_tensor = torch.tensor(x, dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(x_tensor)               # (1, num_classes)
            probs = F.softmax(logits, dim=1)[0]         # (num_classes,)

        # 取 Top-N
        top_n = min(top_n, self.cfg.num_classes)
        top_probs, top_ids = torch.topk(probs, top_n)

        results = []
        for label_id, prob in zip(top_ids.tolist(), top_probs.tolist()):
            results.append({
                "id":          label_id,
                "ou":          self.id2label[label_id],
                "probability": f"{prob:.10f}",
            })
        return results


# ─── 快速驗證 ────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = Predictor()

    # 使用舊系統文件中的範例文本
    sample = (
        "檢舉補習班違規,渠反映太麻里鄉太麻里街上「賢儒補習班」"
        "(民眾不知正確地址)，裡面設有安親班、音樂班及補習班，"
        "其使用坪數未達標準，且老師(表示是主任)有體罰學生的行為，"
        "建請查處。"
    )

    results = predictor.predict(sample, top_n=3)
    print(f"\n輸入：{sample[:40]}...")
    print("\nTop-3 預測結果：")
    for r in results:
        print(f"  [{r['id']}] {r['ou']:<12} {float(r['probability']):.4f}")
