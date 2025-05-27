from typing import List, Dict, Any
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Danh sách các trường sử dụng mô hình Thay thế (TF-IDF)
REPLACEMENT_FIELDS = {'HoTenChucVuNguoiKy'}

class FieldModel:
    def __init__(self, field_code: str, model_type: str = 'embedding'):
        self.field_code = field_code
        self.model_type = model_type  # 'embedding' hoặc 'replacement'
        if model_type == 'embedding':
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
        else:
            self.vectorizer = TfidfVectorizer()
            self.knn = KNeighborsClassifier(n_neighbors=1)
        self.labels = []

    def train(self, ocr_values: List[str], human_values: List[str]):
        if len(ocr_values) < 1 or len(set(human_values)) < 1:
            logging.warning(f"Bỏ qua field {self.field_code}: Không đủ dữ liệu hoặc nhãn.")
            return False
        
        self.labels = list(set(human_values))
        if self.model_type == 'embedding':
            # Tạo embedding cho ocr_values
            X = self.sbert_model.encode(ocr_values, convert_to_tensor=False)
            y = human_values
            try:
                self.knn.fit(X, y)
                logging.info(f"Đã huấn luyện KNN (Embedding) cho field {self.field_code} với {len(ocr_values)} mẫu.")
                return True
            except Exception as e:
                logging.error(f"Lỗi khi huấn luyện KNN (Embedding) cho field {self.field_code}: {e}")
                return False
        else:
            # Sử dụng TF-IDF
            X = self.vectorizer.fit_transform(ocr_values).toarray()
            y = human_values
            try:
                self.knn.fit(X, y)
                logging.info(f"Đã huấn luyện KNN (TF-IDF) cho field {self.field_code} với {len(ocr_values)} mẫu.")
                return True
            except Exception as e:
                logging.error(f"Lỗi khi huấn luyện KNN (TF-IDF) cho field {self.field_code}: {e}")
                return False

    def predict(self, ocr_value: str) -> tuple[str, float]:
        try:
            if self.model_type == 'embedding':
                X = self.sbert_model.encode([ocr_value], convert_to_tensor=False)
                confidence = self.knn.predict_proba(X)[0].max()
                prediction = self.knn.predict(X)[0]
            else:
                X = self.vectorizer.transform([ocr_value]).toarray()
                confidence = self.knn.predict_proba(X)[0].max()
                prediction = self.knn.predict(X)[0]
            return prediction, confidence
        except Exception as e:
            logging.error(f"Lỗi khi dự đoán cho field {self.field_code}: {e}")
            return ocr_value, 0.0

def train_field_models(corrected_docs: List[Dict]) -> Dict[str, FieldModel]:
    field_models = {}
    for doc in corrected_docs:
        for field in doc['fields']:
            field_code = field['doctypefieldcode']
            ocr_value = field['ocr_value']
            human_value = field['corrected_human_value']
            
            if field_code not in field_models:
                model_type = 'replacement' if field_code in REPLACEMENT_FIELDS else 'embedding'
                field_models[field_code] = FieldModel(field_code, model_type)
            
            # Thu thập dữ liệu huấn luyện
            if not hasattr(field_models[field_code], 'train_data'):
                field_models[field_code].train_data = {'ocr_values': [], 'human_values': []}
            field_models[field_code].train_data['ocr_values'].append(ocr_value)
            field_models[field_code].train_data['human_values'].append(human_value)
    
    # Huấn luyện các mô hình
    for field_code, model in field_models.items():
        if hasattr(model, 'train_data'):
            model.train(model.train_data['ocr_values'], model.train_data['human_values'])
            del model.train_data  # Xóa dữ liệu tạm để tiết kiệm bộ nhớ
    
    return field_models

def predict_with_field_models(fields: List[Dict], field_models: Dict[str, FieldModel]) -> List[tuple[str, float]]:
    predictions = []
    for field in fields:
        field_code = field['doctypefieldcode']
        ocr_value = field['ocr_value']
        
        if field_code in field_models:
            prediction, confidence = field_models[field_code].predict(ocr_value)
            predictions.append((prediction, confidence))
        else:
            predictions.append((ocr_value, 0.0))
    
    return predictions

def check_and_retrain_field_models(field_models: Dict[str, FieldModel], corrected_docs: List[Dict], existing_labels: set):
    new_labels = set()
    for doc in corrected_docs:
        for field in doc['fields']:
            new_labels.add(field['corrected_human_value'])
    
    if new_labels - existing_labels:
        logging.info("Phát hiện nhãn mới, huấn luyện lại các mô hình...")
        new_models = train_field_models(corrected_docs)
        field_models.update(new_models)