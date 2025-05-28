import pandas as pd

def load_data(csv_path):
    """
    Load và làm sạch dữ liệu từ file CSV.
    """
    df = pd.read_csv(csv_path, dtype=str)
    df = df.dropna(subset=['docid', 'doctypefieldcode', 'ocr_value', 'human_value'])
    df['ocr_value'] = df['ocr_value'].astype(str).str.strip()
    df['human_value'] = df['human_value'].astype(str).str.strip()
    return df

def compute_statistics(df):
    """
    Thống kê:
    - Số lượng tài liệu (docid duy nhất)
    - Bảng thống kê theo trường thông tin:
        * Mã trường (doctypefieldcode)
        * Số lượng bản ghi
        * Số lượng OCR đúng
        * Số lượng OCR sai
        * Tỷ lệ đúng / sai (%)
    """
    total_docs = df['docid'].nunique()

    df['ocr_correct'] = df['ocr_value'] == df['human_value']

    summary = (
        df.groupby('doctypefieldcode')
        .agg(
            total_records=('ocr_correct', 'count'),
            correct_ocr=('ocr_correct', 'sum'),
        )
        .reset_index()
    )

    summary['incorrect_ocr'] = summary['total_records'] - summary['correct_ocr']
    summary['accuracy (%)'] = (summary['correct_ocr'] / summary['total_records'] * 100).round(2)
    summary['error_rate (%)'] = (summary['incorrect_ocr'] / summary['total_records'] * 100).round(2)
    summary = summary.sort_values(by='error_rate (%)', ascending=False)
   
    return total_docs, summary

def print_statistics(total_docs, summary_df):
    print(f"🧾 Tổng số tài liệu (docid duy nhất): {total_docs}\n")
    print("📊 Bảng thống kê theo trường thông tin:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    CSV_PATH = "sample_ocr_human_value_kiengiang_khaisinh_100K_doc.csv"  # Đổi tên file nếu cần
    df = load_data(CSV_PATH)
    total_docs, summary_df = compute_statistics(df)
    print_statistics(total_docs, summary_df)
