"""
BERT 預測模組。

【白話說明】
和 TextCNN 版相同的介面，但後端用 BERT。
初始化載入一次模型 + tokenizer，之後 predict() 可快速呼叫。
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_loader import build_label_map


class Predictor:

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 載入標籤
        _, self.id2label = build_label_map(self.cfg.labels_path)

        # 載入 tokenizer + 模型
        model_dir = str(self.cfg.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        print(f"[predict] BERT 模型載入完成：{model_dir}")

    def predict(self, text: str, top_n: int = 3) -> list[dict]:
        """
        預測文本分類，回傳 Top-N 結果。
        API 介面和 TextCNN 版完全相同。
        """
        inputs = self.tokenizer(
            text,
            max_length=self.cfg.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)[0]

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


if __name__ == "__main__":
    predictor = Predictor()
    sample = "檢舉補習班違規,渠反映太麻里鄉太麻里街上「賢儒補習班」建請查處。"
    results = predictor.predict(sample, top_n=3)
    print(f"\n輸入：{sample[:40]}...")
    print("\nTop-3 預測結果：")
    for r in results:
        print(f"  [{r['id']}] {r['ou']:<12} {float(r['probability']):.4f}")
