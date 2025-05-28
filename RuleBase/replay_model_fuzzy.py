import pandas as pd
from tqdm import tqdm
from rapidfuzz import process, fuzz

CSV_PATH = "sample_ocr_human_value_kiengiang_khaisinh.csv"
FIELD_FILTER = "GioiTinh"
THRESHOLD = 85

class FuzzyFixer:
    def __init__(self):
        self.human_values = {}

    def learn(self, field, human_value):
        if not human_value:
            return
        self.human_values.setdefault(field, set()).add(human_value.strip())

    def suggest(self, field, ocr_value):
        candidates = list(self.human_values.get(field, []))
        if not candidates or not ocr_value:
            return None
        match = process.extractOne(ocr_value.strip(), candidates, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= THRESHOLD:
            return match[0]
        return None

def run_fuzzy_replay():
    df = pd.read_csv(CSV_PATH)
    df = df[df['doctypefieldcode'] == FIELD_FILTER].dropna(subset=['ocr_value', 'human_value'])
    df['ocr_value'] = df['ocr_value'].astype(str).str.strip()
    df['human_value'] = df['human_value'].astype(str).str.strip()

    fixer = FuzzyFixer()
    correct = 0

    print(f"🚀 Chạy thử nghiệm Fuzzy Matching (Levenshtein) trên trường '{FIELD_FILTER}' với {len(df)} bản ghi...\n")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="🔁 Replay", unit="record"):
        field = row['doctypefieldcode']
        ocr = row['ocr_value']
        human = row['human_value']

        suggestion = fixer.suggest(field, ocr)
        if suggestion == human:
            correct += 1

        fixer.learn(field, human)

    accuracy = round(100 * correct / len(df), 2)
    print(f"✅ Fuzzy Matching Accuracy: {accuracy}% ({correct}/{len(df)})")

if __name__ == "__main__":
    run_fuzzy_replay()
