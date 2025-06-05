# Module: Error Correction (Rule-based + ML-based)
# Author: ChatGPT
# Description: A lightweight module to correct OCR errors using combined rule-based and ML-based strategies.

import re
import json
import string
import unicodedata
import random
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import joblib
import os
from synthetic_error_generator import generate_synthetic_errors

# --------------------- Preprocessing ---------------------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# --------------------- Rule-Based Suggestion ---------------------
def rule_based_correction(ocr: str, memory: List[Tuple[str, str]]) -> Optional[str]:
    ocr_norm = normalize_text(ocr)
    for old_ocr, human_val in memory[::-1]:  # prioritize latest entries
        if normalize_text(old_ocr) in ocr_norm or ocr_norm in normalize_text(old_ocr):
            return human_val
    return None

# --------------------- ML-Based Correction ---------------------
class CorrectionModel:
    def __init__(self, model_path: str = "correction_model.pkl"):
        self.model_path = model_path
        self.pipeline = None

    def train(self, data: List[Tuple[str, str]]):
        X = [normalize_text(x[0]) for x in data]
        y = [x[1] for x in data]
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('knn', KNeighborsClassifier(n_neighbors=1))
        ])
        self.pipeline.fit(X, y)
        joblib.dump(self.pipeline, self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)

    def predict(self, ocr_value: str) -> Optional[str]:
        if not self.pipeline:
            return None
        try:
            ocr_norm = normalize_text(ocr_value)
            return self.pipeline.predict([ocr_norm])[0]
        except Exception:
            return None

# --------------------- Main Correction Module ---------------------
class CorrectionEngine:
    def __init__(self):
        self.memory: List[Tuple[str, str]] = []
        self.model = CorrectionModel()
        self.model.load()

    def add_training_data(self, ocr: str, human: str):
        if normalize_text(ocr) != normalize_text(human):
            self.memory.append((ocr, human))

    def train_model(self):
        if self.memory:
            self.model.train(self.memory)

    def suggest(self, ocr_value: str) -> Tuple[Optional[str], str]:
        # Rule-based first
        suggestion = rule_based_correction(ocr_value, self.memory)
        if suggestion:
            return suggestion, "rule"
        # ML-based
        suggestion = self.model.predict(ocr_value)
        if suggestion:
            return suggestion, "ml"
        return None, "none"

    def interactive_add_and_test(self):
        print("\n=== CHẾ ĐỘ TƯƠNG TÁC: TỰ SINH LỖI & GỢI Ý CHỈNH SỬA ===")
        while True:
            true_value = input("\nNhập giá trị đúng (hoặc 'exit' để thoát): ").strip()
            if true_value.lower() == 'exit':
                break

            synthetic_errors = generate_synthetic_errors(true_value, n_errors=5)
            print(f"Tự sinh {len(synthetic_errors)} giá trị lỗi từ '{true_value}':")
            for err in synthetic_errors:
                print(f" - {err}")
                self.add_training_data(err, true_value)

            self.train_model()

            while True:
                ocr_test = input("\nNhập giá trị OCR để kiểm tra (hoặc 'back' để quay lại): ").strip()
                if ocr_test.lower() == 'back':
                    break
                predicted, method = self.suggest(ocr_test)
                print(f"[→] Dự đoán: '{ocr_test}' → '{predicted}' (source: {method})")

# --------------------- Testing Example ---------------------
if __name__ == "__main__":
    engine = CorrectionEngine()

    # Simulate first document corrections
    training_pairs = [
        ("Can cươc cong dan", "Căn cước công dân"),
        ("Đinh Quang Huy - PCT", "Đinh Quang Huy"),
        ("Nguyễn văn B", "Nguyễn Văn B")
    ]
    for ocr, human in training_pairs:
        engine.add_training_data(ocr, human)

    # Train model
    engine.train_model()

    # Simulate second document inference
    test_ocr = [
        "Căn cuac cộng đn",
        "Dinn Quano Huu PcT",
        "Nguyen Van B"
    ]
    for test in test_ocr:
        suggestion, source = engine.suggest(test)
        print(f"OCR: {test} -> Suggested: {suggestion} (source: {source})")

    # Interactive test with new label
    engine.interactive_add_and_test()
