import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm
import unicodedata
import re

CSV_PATH = "sample_ocr_human_value_kiengiang_khaisinh.csv"
FIELD_FILTER = "NoiSinh"  # Trường cần lọc
THRESHOLD = 0.85  # Ngưỡng cosine tương đương khoảng cách L2 nhỏ
DIM = 384  # mặc định cho MiniLM, SBERT là 768

def normalize_text(text):
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

class EmbeddingFixer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = faiss.IndexFlatL2(768 if "sbert" in model_name else 384)
        self.human_values = []

    def encode(self, text):
        vec = self.model.encode([normalize_text(text)])[0].astype(np.float32)
        faiss.normalize_L2(vec.reshape(1, -1))
        return vec

    def learn(self, human_value):
        if not human_value:
            return
        vec = self.encode(human_value)
        self.index.add(np.array([vec]))
        self.human_values.append(human_value)

    def suggest(self, ocr_value):
        if not self.human_values:
            return None
        vec = self.encode(ocr_value)
        D, I = self.index.search(np.array([vec]), k=1)
        if D[0][0] < (2 - 2 * THRESHOLD):  # cosine similarity to L2 distance
            return self.human_values[I[0][0]]
        return None

def run_comparison():
    df = pd.read_csv(CSV_PATH)
    df = df[df['doctypefieldcode'] == FIELD_FILTER].dropna(subset=['ocr_value', 'human_value'])
    df['ocr_value'] = df['ocr_value'].astype(str).str.strip()
    df['human_value'] = df['human_value'].astype(str).str.strip()

    models = {
        "MiniLM": EmbeddingFixer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        "ViSBERT": EmbeddingFixer("keepitreal/vietnamese-sbert")
    }

    results = {model: [] for model in models}

    print(f"🚀 Chạy lại mô hình FAISS với normalize + IndexFlatL2 trên '{FIELD_FILTER}' với {len(df)} bản ghi...\\n")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔁 Replay", unit="record"):
        ocr = row['ocr_value']
        human = row['human_value']

        for name, fixer in models.items():
            suggestion = fixer.suggest(ocr)
            is_correct = suggestion == human if suggestion else False
            results[name].append(is_correct)
            fixer.learn(human)

    for name in models:
        correct = sum(results[name])
        total = len(results[name])
        accuracy = round(100 * correct / total, 2)
        print(f"✅ Model {name} Accuracy: {accuracy}% ({correct}/{total})")

if __name__ == "__main__":
    run_comparison()