import pandas as pd
import os
from typing import List, Dict

def load_replay_data(file_path: str = "replay/sample_ocr_human_value.csv") -> List[Dict]:
    """
    Đọc dữ liệu từ file CSV và nhóm theo docid.
    :param file_path: Đường dẫn tới file CSV
    :return: Danh sách các tài liệu, mỗi tài liệu là một dict chứa danh sách trường thông tin
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    
    df = pd.read_csv(file_path)
    grouped = df.groupby('docid')
    documents = []
    
    for docid, group in grouped:
        doc = {
            'docid': docid,
            'fields': [
                {
                    'doctypefieldcode': row['doctypefieldcode'],
                    'ocr_value': row['ocr_value'],
                    'human_value': row['human_value']
                }
                for _, row in group.iterrows()
            ]
        }
        documents.append(doc)
    
    return documents

def save_corrected_data(data: List[Dict], file_path: str = "replay/corrected_data.csv"):
    """
    Lưu dữ liệu đã sửa vào file CSV.
    :param data: Danh sách dữ liệu đã sửa
    :param file_path: Đường dẫn lưu file
    """
    rows = []
    for doc in data:
        for field in doc['fields']:
            rows.append({
                'docid': doc['docid'],
                'doctypefieldcode': field['doctypefieldcode'],
                'ocr_value': field['ocr_value'],
                'human_value': field.get('corrected_human_value', field['human_value'])
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)