import pandas as pd
from collections import defaultdict
from rapidfuzz import fuzz, process

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


# === EVALUATION FUNCTION ===
def evaluate_predictions(df, predictions):
    total = len(df)
    correct = sum(df['human_value'] == predictions)
    accuracy = round((correct / total) * 100, 2) if total > 0 else 0
    return accuracy


# === MAIN REPLAY FUNCTION ===
def run_replay(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['doctypefieldcode', 'ocr_value', 'human_value'])
    fields = df['doctypefieldcode'].unique()

    memory_fixer = MemoryBasedFixer()
    fassi_fixer = FassiLikeFixer()

    results = []
    predict_results = []
    for field in fields:
        df_field = df[df['doctypefieldcode'] == field].copy()

        mem_preds = []
        fassi_preds = []

        for _, row in df_field.iterrows():
            ocr = row['ocr_value']
            human = row['human_value']

            # Predict before learn
            mem_pred = memory_fixer.suggest(field, ocr)
            fassi_pred = fassi_fixer.suggest(field, ocr)

            mem_preds.append(mem_pred)
            fassi_preds.append(fassi_pred)

            # Learn after predict
            memory_fixer.learn(field, ocr, human)
            fassi_fixer.learn(field, ocr, human)

            # Store predictions in the DataFrame
            predict_results.append({
                'Field': field,
                'OCR_Value': ocr,
                'Human_Value': human,
                'MemoryBased_Prediction': mem_pred,
                'FASSI_Prediction': fassi_pred,
                'MemoryBased_Prediction_correct':'True' if mem_pred == human else 'False',
                'FASSI_Prediction_corect': 'True' if fassi_pred == human else 'False'
            })

        df_field['mem_predict'] = mem_preds
        df_field['fassi_predict'] = fassi_preds

        mem_acc = evaluate_predictions(df_field, df_field['mem_predict'])
        fassi_acc = evaluate_predictions(df_field, df_field['fassi_predict'])

        results.append({
            'Field': field,
            'MemoryBased_Accuracy (%)': mem_acc,
            'FASSI_Accuracy (%)': fassi_acc
        })

    result_df = pd.DataFrame(results).sort_values(by='FASSI_Accuracy (%)', ascending=False)
    print("===== KẾT QUẢ ĐÁNH GIÁ THEO TRƯỜNG THÔNG TIN =====")
    print(result_df.to_string(index=False))

    # Export CSV
    result_df.to_csv("field_accuracy_comparison.csv", index=False)
    # export prediction_results to CSV
    predict_df = pd.DataFrame(predict_results)
    predict_df.to_csv("field_prediction_results.csv", index=False)  


# === RUN ===
if __name__ == "__main__":
    run_replay(CSV_PATH)
