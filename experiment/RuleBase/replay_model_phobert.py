import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import unicodedata
import re
import torch
import time
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict

CSV_PATH = "sample_ocr_human_value_kiengiang_khaisinh.csv"
FIELD_FILTER = "NoiSinh"
THRESHOLD = 0.85

def normalize_text(text):
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # (batch_size, seq_len, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

class EmbeddingFixer:
    def __init__(self, model_name, dim, device=None):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.human_values = []

        print(f"Init model {model_name} on {self.device}")

        if "sentence-transformers" in model_name:
            # Model riêng của SentenceTransformer
            self.st_model = SentenceTransformer(model_name, device=self.device)
            self.uses_sentence_transformer = True
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.uses_sentence_transformer = False

    def encode(self, text):
        norm_text = normalize_text(text)
        if self.uses_sentence_transformer:
            vec = self.st_model.encode([norm_text], normalize_embeddings=True)[0]
            return np.array(vec, dtype=np.float32)
        else:
            encoded_input = self.tokenizer(norm_text, return_tensors='pt', truncation=True, max_length=128)
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            embedding = mean_pooling(model_output, encoded_input['attention_mask'])
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            vec = embedding.cpu().numpy()[0].astype(np.float32)
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
        cosine_sim = 1 - D[0][0] / 2  # L2 distance to cosine sim
        if cosine_sim >= THRESHOLD:
            return self.human_values[I[0][0]]
        return None

def run_comparison():
    df = pd.read_csv(CSV_PATH)
    df = df[df['doctypefieldcode'] == FIELD_FILTER].dropna(subset=['ocr_value', 'human_value'])
    df['ocr_value'] = df['ocr_value'].astype(str).str.strip()
    df['human_value'] = df['human_value'].astype(str).str.strip()

    models = {
        "PhoBERT-base": EmbeddingFixer("vinai/phobert-base", dim=768),
        "ViSBERT": EmbeddingFixer("keepitreal/vietnamese-sbert", dim=768),
        "SimCSE-PhoBERT": EmbeddingFixer("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base", dim=768),
        "MiniLM": EmbeddingFixer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", dim=384),
    }

    results = {name: [] for name in models}
    times = defaultdict(list)

    print(f"🚀 Bắt đầu chạy so sánh trên trường '{FIELD_FILTER}', số lượng bản ghi: {len(df)}...\n")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔁 Replay", unit="record"):
        ocr = row['ocr_value']
        human = row['human_value']

        for name, fixer in models.items():
            start = time.time()
            suggestion = fixer.suggest(ocr)
            duration = time.time() - start
            times[name].append(duration)

            is_correct = suggestion == human if suggestion else False
            results[name].append(is_correct)
            fixer.learn(human)

    for name in models:
        correct = sum(results[name])
        total = len(results[name])
        accuracy = round(100 * correct / total, 2)
        min_time = round(min(times[name]) * 1000, 2)
        max_time = round(max(times[name]) * 1000, 2)
        avg_time = round(sum(times[name]) / len(times[name]) * 1000, 2)
        print(f"\n✅ Model {name}")
        print(f"   - Accuracy   : {accuracy}% ({correct}/{total})")
        print(f"   - Time [ms]  : Min={min_time}, Max={max_time}, Avg={avg_time}")

if __name__ == "__main__":
    run_comparison()
