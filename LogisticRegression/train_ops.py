from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
import numpy as np
from typing import List, Dict, Tuple

class OCRCorrectionModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.knn = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.X_train = None
        self.y_train = None
        self.labels = set()
    
    def train(self, data: List[Dict]):
        """
        Huấn luyện model KNN trên dữ liệu đã sửa.
        :param data: Danh sách các trường với OCR và human value
        """
        print(f"Log: Bắt đầu huấn luyện KNN với {len(data)} mẫu")
        X = []
        y = []
        for field in data:
            X.append(field['ocr_value'])
            y.append(field.get('corrected_human_value', field['human_value']))
            self.labels.add(field['doctypefieldcode'])
        
        if len(X) < 1:
            print(f"Log: Không đủ mẫu để huấn luyện KNN (chỉ có {len(X)} mẫu)")
            return
        
        try:
            # Chuyển đổi văn bản thành vector TF-IDF
            X_tfidf = self.vectorizer.fit_transform(X)
            self.knn.fit(X_tfidf)
            self.X_train = X_tfidf
            self.y_train = y
            print(f"Log: Huấn luyện KNN thành công với {len(X)} mẫu, labels: {self.labels}")
        except Exception as e:
            print(f"Log: Lỗi huấn luyện KNN: {e}")
            raise e
    
    def predict(self, ocr_values: List[str]) -> List[Tuple[str, float]]:
        """
        Dự đoán giá trị human_value và độ tin cậy bằng KNN.
        :param ocr_values: Danh sách giá trị OCR
        :return: Danh sách tuple (predicted_value, confidence)
        """
        print(f"Log: Bắt đầu dự đoán KNN với {len(ocr_values)} giá trị OCR")
        if not ocr_values or self.X_train is None:
            print("Log: Không có giá trị OCR hoặc chưa huấn luyện")
            return [("", 0.0) for _ in ocr_values]
        
        try:
            X_test = self.vectorizer.transform(ocr_values)
            distances, indices = self.knn.kneighbors(X_test)
            predictions = [self.y_train[idx[0]] for idx in indices]
            # Confidence dựa trên khoảng cách cosine (1 - distance)
            confidences = [1 - dist[0] if dist[0] < 1 else 0.0 for dist in distances]
            print(f"Log: Dự đoán KNN thành công, số kết quả: {len(predictions)}")
            return list(zip(predictions, confidences))
        except Exception as e:
            print(f"Log: Lỗi dự đoán KNN: {e}")
            return [("", 0.0) for _ in ocr_values]

def train_field_models(data: List[Dict]) -> Dict[str, OCRCorrectionModel]:
    """
    Huấn luyện model KNN riêng cho từng doctypefieldcode.
    :param data: Danh sách các tài liệu
    :return: Dictionary chứa model cho từng field code
    """
    print("Log: Bắt đầu train_field_models")
    field_data = {}
    for doc in data:
        for field in doc['fields']:
            field_code = field['doctypefieldcode']
            if field_code not in field_data:
                field_data[field_code] = []
            field_data[field_code].append(field)
    
    field_models = {}
    for field_code, fields in field_data.items():
        print(f"Log: Huấn luyện KNN cho field {field_code} với {len(fields)} mẫu")
        model = OCRCorrectionModel()
        model.train(fields)
        if model.labels and model.X_train is not None:  # Chỉ thêm model nếu huấn luyện thành công
            field_models[field_code] = model
            print(f"Log: Đã thêm model KNN cho field {field_code}")
        else:
            print(f"Log: Bỏ qua field {field_code} do không huấn luyện được")
    
    print(f"Log: Kết thúc train_field_models, số model: {len(field_models)}")
    return field_models

def predict_with_field_models(fields: List[Dict], field_models: Dict[str, OCRCorrectionModel]) -> List[Tuple[str, float]]:
    """
    Dự đoán cho các trường sử dụng model KNN tương ứng.
    :param fields: Danh sách các trường trong tài liệu
    :param field_models: Dictionary chứa model cho từng field code
    :return: Danh sách tuple (predicted_value, confidence)
    """
    print(f"Log: Bắt đầu predict_with_field_models cho {len(fields)} fields")
    predictions = []
    for field in fields:
        field_code = field['doctypefieldcode']
        ocr_value = field['ocr_value']
        print(f"Log: Dự đoán cho field {field_code}, OCR: {ocr_value}")
        if field_code in field_models:
            try:
                pred = field_models[field_code].predict([ocr_value])
                predictions.append(pred[0] if pred else ("", 0.0))
                print(f"Log: Dự đoán thành công cho field {field_code}")
            except Exception as e:
                print(f"Log: Lỗi dự đoán cho field {field_code}: {e}")
                predictions.append(("", 0.0))
        else:
            print(f"Log: Không có model cho field {field_code}")
            predictions.append(("", 0.0))
    print(f"Log: Kết thúc predict_with_field_models, số dự đoán: {len(predictions)}")
    return predictions

def check_and_retrain_field_models(field_models: Dict[str, OCRCorrectionModel], data: List[Dict], existing_labels: set):
    """
    Kiểm tra nhãn mới và huấn luyện bổ sung.
    :param field_models: Dictionary chứa model cho từng field code
    :param data: Dữ liệu mới
    :param existing_labels: Tập hợp nhãn đã huấn luyện
    """
    print("Log: Bắt đầu check_and_retrain_field_models")
    field_data = {}
    for doc in data:
        for field in doc['fields']:
            field_code = field['doctypefieldcode']
            if field_code not in field_data:
                field_data[field_code] = []
            field_data[field_code].append(field)
    
    for field_code, fields in field_data.items():
        print(f"Log: Kiểm tra field {field_code} với {len(fields)} mẫu")
        if field_code not in existing_labels:
            print(f"Log: Field {field_code} là nhãn mới, bắt đầu huấn luyện")
            if field_code not in field_models:
                field_models[field_code] = OCRCorrectionModel()
            field_models[field_code].train(fields)
            print(f"Log: Huấn luyện bổ sung thành công cho field {field_code}")
    print("Log: Kết thúc check_and_retrain_field_models")