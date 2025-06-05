import pandas as pd
from collections import defaultdict
from rapidfuzz import fuzz, process
from tqdm import tqdm
from faiss_embedding_fixer import FassisEmbeddingFixer

# === CONFIG ===
CSV_PATH = "sample_ocr_human_value_kiengiang_khaisinh.csv"
FUZZY_SCORE_THRESHOLD = 20

# === MEMORY-BASED FIX MODULE ===
class MemoryBasedFixer:
    def __init__(self):
        self.memory = defaultdict(set)

    def learn(self, field, ocr_value, human_value):
        if pd.notna(human_value):
            self.memory[field].add(human_value.strip())

    def suggest(self, field, ocr_value):
        candidates = list(self.memory[field])
        if not candidates or pd.isna(ocr_value):
            return None
        result = process.extractOne(ocr_value, candidates, scorer=fuzz.token_sort_ratio)
        if result is None:
            return None
        match, score, _ = result
        return match if score >= FUZZY_SCORE_THRESHOLD else None

# === FASSI-LIKE FIX MODULE ===
class FassiLikeFixer:
    def __init__(self):
        self.known_values = defaultdict(lambda: defaultdict(int))

    def learn(self, field, ocr_value, human_value):
        if pd.notna(ocr_value) and pd.notna(human_value):
            self.known_values[field][(ocr_value.strip(), human_value.strip())] += 1

    def suggest(self, field, ocr_value):
        if pd.isna(ocr_value):
            return None
        candidates = [(hval, count) for (oval, hval), count in self.known_values[field].items() if oval == ocr_value.strip()]
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]


# === MAIN REPLAY FUNCTION ===
def run_replay(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['doctypefieldcode', 'ocr_value', 'human_value'])
    df['ocr_value'] = df['ocr_value'].astype(str).str.strip()
    df['human_value'] = df['human_value'].astype(str).str.strip()

    df = df[df['doctypefieldcode'] == 'NoiSinh'].reset_index(drop=True)

    memory_fixer = MemoryBasedFixer()
    #fassi_fixer = FassiLikeFixer()
    print("\n🚀 Đang khởi tạo mô hình Faiss model...")
    fassi_fixer = FassisEmbeddingFixer()


    df['mem_predict'] = ''
    df['fassi_predict'] = ''
    df['ocr_correct'] = df['ocr_value'] == df['human_value']

    print("\n🚀 Đang chạy thử nghiệm mô hình sửa lỗi...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔁 Replay", unit="record"):
        field = row['doctypefieldcode']
        ocr = row['ocr_value']
        human = row['human_value']

        mem_pred = memory_fixer.suggest(field, ocr)
        fassi_pred = fassi_fixer.suggest(field, ocr)

        df.at[idx, 'mem_predict'] = mem_pred if mem_pred else ''
        df.at[idx, 'fassi_predict'] = fassi_pred if fassi_pred else ''

        memory_fixer.learn(field, ocr, human)
        #fassi_fixer.learn(field, ocr, human) # dùng cho class fassi_like_fixer
        if fassi_pred!=human:
            fassi_fixer.learn(field, human)
    
    print("💾 Đang lưu model FAISS...")
    fassi_fixer.save_all()
    
    # Đánh giá kết quả
    df['mem_correct'] = df['mem_predict'] == df['human_value']
    df['fassi_correct'] = df['fassi_predict'] == df['human_value']
    df['mem_suggested'] = df['mem_predict'] != ''
    df['fassi_suggested'] = df['fassi_predict'] != ''

    # Thống kê tổng hợp theo trường
    summary = (
        df.groupby('doctypefieldcode')
        .agg(
            total_records=('ocr_value', 'count'),
            correct_ocr=('ocr_correct', 'sum'),
            mem_correct=('mem_correct', 'sum'),
            fassi_correct=('fassi_correct', 'sum'),
            mem_suggested=('mem_suggested', 'sum'),
            fassi_suggested=('fassi_suggested', 'sum')
        )
        .reset_index()
    )

    summary['incorrect_ocr'] = summary['total_records'] - summary['correct_ocr']
    summary['accuracy (%)'] = (summary['correct_ocr'] / summary['total_records'] * 100).round(2)
    summary['error_rate (%)'] = (summary['incorrect_ocr'] / summary['total_records'] * 100).round(2)
    summary['MemoryBased_Accuracy (%)'] = (summary['mem_correct'] / summary['total_records'] * 100).round(2)
    summary['FASSI_Accuracy (%)'] = (summary['fassi_correct'] / summary['total_records'] * 100).round(2)
    summary['MemoryBased_Suggested (%)'] = (summary['mem_suggested'] / summary['total_records'] * 100).round(2)
    summary['FASSI_Suggested (%)'] = (summary['fassi_suggested'] / summary['total_records'] * 100).round(2)

    # Tính Accuracy tổng thể (dù có suggest hay không)
    df['mem_final_correct'] = df.apply(
        lambda row: row['mem_predict'] == row['human_value'] if row['mem_predict'] else row['ocr_value'] == row['human_value'],
        axis=1
    )
    df['fassi_final_correct'] = df.apply(
        lambda row: row['fassi_predict'] == row['human_value'] if row['fassi_predict'] else row['ocr_value'] == row['human_value'],
        axis=1
    )

    overall_acc = (
        df.groupby('doctypefieldcode')
        .agg(
            mem_overall=('mem_final_correct', lambda x: round(x.mean() * 100, 2)),
            fassi_overall=('fassi_final_correct', lambda x: round(x.mean() * 100, 2))
        )
        .reset_index()
    )

    # Gộp vào bảng summary
    summary = pd.merge(summary, overall_acc, on='doctypefieldcode', how='left')
    summary.rename(columns={
        'mem_overall': 'MemoryBased_Overall_Accuracy (%)',
        'fassi_overall': 'FASSI_Overall_Accuracy (%)'
    }, inplace=True)


    # Sắp xếp và chọn cột
    summary = summary[
        ['doctypefieldcode', 'total_records', 'correct_ocr', 'incorrect_ocr',
         'accuracy (%)', 'error_rate (%)',
         'MemoryBased_Accuracy (%)', 'FASSI_Accuracy (%)',
         'MemoryBased_Suggested (%)', 'FASSI_Suggested (%)','MemoryBased_Overall_Accuracy (%)', 'FASSI_Overall_Accuracy (%)']
    ].sort_values(by='error_rate (%)', ascending=False)

    # Hiển thị và lưu
    print(f"\n🧾 Tổng số bản ghi đã chạy: {len(df)}\n")
    print("📊 BẢNG THỐNG KÊ:")
    print(summary.to_string(index=False))

    summary.to_csv("model_comparison_summary.csv", index=False)
    df.to_csv("model_prediction_full_output.csv", index=False)

# === RUN ===
if __name__ == "__main__":
    run_replay(CSV_PATH)
