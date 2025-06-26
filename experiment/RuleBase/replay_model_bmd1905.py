import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from collections import defaultdict
import torch
import time
# requirements: transformers, torch, pandas, tqdm, sentencepiece
CSV_PATH = "input/sample_ocr_human_value_kiengiang_khaisinh.csv"
FIELD_FILTER = "NoiSinh"

class VietnameseCorrectionModel:
    def __init__(self, model_name="bmd1905/vietnamese-correction-v2", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"Loaded model {model_name} on {self.device}")

    def correct(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=128)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()


def run_correction():
    df = pd.read_csv(CSV_PATH)
    df = df[df['doctypefieldcode'] == FIELD_FILTER].dropna(subset=['ocr_value', 'human_value'])
    df['ocr_value'] = df['ocr_value'].astype(str).str.strip()
    df['human_value'] = df['human_value'].astype(str).str.strip()

    corrector = VietnameseCorrectionModel()
    results = []
    times = []

    print(f"🚀 Bắt đầu chạy correction trên trường '{FIELD_FILTER}', số lượng bản ghi: {len(df)}...\n")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔁 Replay", unit="record"):
        ocr = row['ocr_value']
        human = row['human_value']

        start = time.time()
        suggestion = corrector.correct(ocr)
        duration = time.time() - start
        times.append(duration)

        is_correct = suggestion == human
        results.append(is_correct)

    # Summary
    correct = sum(results)
    total = len(results)
    accuracy = round(100 * correct / total, 2)
    min_time = round(min(times) * 1000, 2)
    max_time = round(max(times) * 1000, 2)
    avg_time = round(sum(times) / len(times) * 1000, 2)

    print(f"\n✅ Model: bmd1905/vietnamese-correction-v2")
    print(f"   - Accuracy   : {accuracy}% ({correct}/{total})")
    print(f"   - Time [ms]  : Min={min_time}, Max={max_time}, Avg={avg_time}")

if __name__ == "__main__":
    run_correction()