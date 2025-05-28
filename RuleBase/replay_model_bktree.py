import pandas as pd
from tqdm import tqdm
from pybktree import BKTree
from Levenshtein import distance as levenshtein_distance
CSV_PATH = "sample_ocr_human_value_kiengiang_khaisinh.csv"
FIELD_FILTER = "GioiTinh"  # Trường cần lọc
MAX_DISTANCE = 2  # Khoảng cách tối đa cho Levenshtein

class BKTreeFixer:
    def __init__(self):
        self.trees = {}
        self.value_sets = {}

    def learn(self, field, human_value):
        if not human_value:
            return
        if field not in self.trees:
            self.value_sets[field] = set()
        if human_value not in self.value_sets[field]:
            self.value_sets[field].add(human_value)
            self.trees[field] = BKTree(levenshtein_distance, list(self.value_sets[field]))

    def suggest(self, field, ocr_value):
        if field not in self.trees or not ocr_value:
            return None
        matches = self.trees[field].find(ocr_value, MAX_DISTANCE)
        if matches:
            matches.sort(key=lambda x: (x[0], x[1]))  # sort by distance then value
            return matches[0][1]
        return None

def run_bktree_replay():
    df = pd.read_csv(CSV_PATH)
    df = df[df['doctypefieldcode'] == FIELD_FILTER].dropna(subset=['ocr_value', 'human_value'])
    df['ocr_value'] = df['ocr_value'].astype(str).str.strip()
    df['human_value'] = df['human_value'].astype(str).str.strip()

    fixer = BKTreeFixer()
    correct = 0

    print(f"🚀 Đang chạy thử nghiệm BK-Tree trên trường '{FIELD_FILTER}' với {len(df)} bản ghi...\n")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="🔁 Replay", unit="record"):
        field = row['doctypefieldcode']
        ocr = row['ocr_value']
        human = row['human_value']

        suggestion = fixer.suggest(field, ocr)
        if suggestion == human:
            correct += 1

        fixer.learn(field, human)

    accuracy = round(100 * correct / len(df), 2)
    print(f"✅ BK-Tree Accuracy: {accuracy}% ({correct}/{len(df)})")

if __name__ == "__main__":
    run_bktree_replay()
