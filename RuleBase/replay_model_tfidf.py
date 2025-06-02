import pandas as pd
import numpy as np
import time
import os
import unicodedata
import re
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CSV_PATH = "input/sample_ocr_human_value_kiengiang_khaisinh.csv"
THRESHOLD = 0.85
CONFIDENCE_FALLBACK = 0.8
FIELD_FILTER = None
MIN_OCCURRENCE = 2

def normalize_text(text):
    text = str(text).lower()
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class TfidfFixer:
    def __init__(self):
        self.human_memory = defaultdict(list)
        self.human_memory_set = defaultdict(set)
        self.human_counter = defaultdict(lambda: defaultdict(int))  # human_value counter
        self.vectorizers = {}
        self.vector_matrices = {}
        self.lookup_dict = defaultdict(dict)  # field -> ocr_value -> last human_value

    def learn_if_valid(self, field, ocr_value, human_value, suggestion):
        if not human_value:
            return

        # Cập nhật lookup_dict
        self.lookup_dict[field][ocr_value] = human_value

        # Cập nhật bộ đếm
        self.human_counter[field][human_value] += 1

        # Chỉ học nếu human_value lặp lại >= MIN_OCCURRENCE và chưa học, và suggest != human
        if (
            self.human_counter[field][human_value] >= MIN_OCCURRENCE
            and human_value not in self.human_memory_set[field]
            and suggestion != human_value
        ):
            self.human_memory[field].append(human_value)
            self.human_memory_set[field].add(human_value)
            self._update_vectorizer(field)

    def _update_vectorizer(self, field):
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        mat = vectorizer.fit_transform(self.human_memory[field])
        self.vectorizers[field] = vectorizer
        self.vector_matrices[field] = mat

    def suggest(self, field, ocr_value):
        if not self.human_memory[field]:
            return None, 0.0
        vec = self.vectorizers[field].transform([ocr_value])
        sim = cosine_similarity(vec, self.vector_matrices[field])[0]
        best_idx = np.argmax(sim)
        best_score = sim[best_idx]
        return self.human_memory[field][best_idx] if best_score >= THRESHOLD else None, round(float(best_score), 4)

    def fallback_lookup(self, field, ocr_value):
        return self.lookup_dict[field].get(ocr_value, None)

def run_replay(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["doctypefieldcode", "ocr_value", "human_value"])
    df["ocr_value"] = df["ocr_value"].astype(str)
    df["human_value"] = df["human_value"].astype(str)
    df["ocr_norm"] = df["ocr_value"].apply(normalize_text)

    if FIELD_FILTER:
        df = df[df["doctypefieldcode"] == FIELD_FILTER].reset_index(drop=True)

    tfidf_fixer = TfidfFixer()
    df["ocr_correct"] = df["ocr_value"].str.strip() == df["human_value"].str.strip()
    df["tfidf_predict"] = ""
    df["tfidf_confidence"] = 0.0
    df["tfidf_correct"] = False
    df["tfidf_suggested"] = False
    df["tfidf_final_correct"] = False
    df["predict_time_ms"] = 0.0

    print("\n🚀 Đang chạy TF-IDF Replay...\n")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔁 Replay", unit="record"):
        try:
            field = row["doctypefieldcode"]
            ocr_raw = row["ocr_value"]
            ocr = row["ocr_norm"]
            human = row["human_value"]

            start = time.time()
            suggestion, confidence = tfidf_fixer.suggest(field, ocr)

            # Fallback nếu confidence < CONFIDENCE_FALLBACK
            if confidence < CONFIDENCE_FALLBACK or suggestion is None:
                fallback = tfidf_fixer.fallback_lookup(field, ocr_raw)
                if fallback:
                    suggestion = fallback
            elapsed = (time.time() - start) * 1000

            df.at[idx, "tfidf_predict"] = suggestion if suggestion else ""
            df.at[idx, "tfidf_confidence"] = confidence
            df.at[idx, "tfidf_suggested"] = bool(suggestion)
            df.at[idx, "tfidf_correct"] = (suggestion == human or suggestion == "")
            df.at[idx, "tfidf_final_correct"] = (suggestion == human) if suggestion else (ocr_raw == human)
            df.at[idx, "predict_time_ms"] = elapsed

            if not row["ocr_correct"]:
                tfidf_fixer.learn_if_valid(field, ocr_raw, human, suggestion)

        except Exception as e:
            print(f"⚠️  Error at row {idx}: {e}")

    print("\n📊 Đang tổng hợp kết quả...\n")
    summary = (
        df.groupby("doctypefieldcode")
        .agg(
            total_records=("ocr_value", "count"),
            correct_ocr=("ocr_correct", "sum"),
            tfidf_predicted=("tfidf_suggested", "sum"),
            tfidf_predicted_correct=("tfidf_correct", "sum"),
            tfidf_final_correct=("tfidf_final_correct", "sum"),
            min_time_ms=("predict_time_ms", "min"),
            max_time_ms=("predict_time_ms", "max"),
            avg_time_ms=("predict_time_ms", "mean")
        )
        .reset_index()
    )

    summary["incorrect_ocr"] = summary["total_records"] - summary["correct_ocr"]
    summary["tfidf_predicted_wrong"] = summary["tfidf_predicted"] - summary["tfidf_predicted_correct"]
    summary["accuracy (%)"] = (summary["correct_ocr"] / summary["total_records"] * 100).round(2)
    summary["error_rate (%)"] = (summary["incorrect_ocr"] / summary["total_records"] * 100).round(2)
    summary["TFIDF_Predicted (%)"] = (summary["tfidf_predicted"] / summary["total_records"] * 100).round(2)
    summary["TFIDF_Predict_Accuracy (%)"] = (
        summary["tfidf_predicted_correct"] / summary["tfidf_predicted"].replace(0, np.nan) * 100
    ).fillna(0).round(2)
    summary["TFIDF_Predict_Error (%)"] = (
        summary["tfidf_predicted_wrong"] / summary["tfidf_predicted"].replace(0, np.nan) * 100
    ).fillna(0).round(2)
    summary["TFIDF_Overall_Accuracy (%)"] = (summary["tfidf_final_correct"] / summary["total_records"] * 100).round(2)
    summary["min_time_ms"] = summary["min_time_ms"].round(2)
    summary["max_time_ms"] = summary["max_time_ms"].round(2)
    summary["avg_time_ms"] = summary["avg_time_ms"].round(2)

    summary = summary[
        [
            "doctypefieldcode", "total_records", "correct_ocr", "incorrect_ocr",
            "accuracy (%)", "error_rate (%)",
            "tfidf_predicted", "TFIDF_Predicted (%)",
            "tfidf_predicted_correct", "tfidf_predicted_wrong",
            "TFIDF_Predict_Accuracy (%)", "TFIDF_Predict_Error (%)",
            "TFIDF_Overall_Accuracy (%)",
            "min_time_ms", "max_time_ms", "avg_time_ms"
        ]
    ].sort_values(by="error_rate (%)", ascending=False)

    os.makedirs("result", exist_ok=True)
    df.to_csv("result/tfidf_prediction_full_output.csv", index=False)
    summary.to_csv("result/tfidf_model_summary.csv", index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    run_replay(CSV_PATH)