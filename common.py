from pydantic import BaseModel
from typing import List
from model.tfidf.tfidf import TfidfModel
# Giả định có KNNModel, thay thế bằng import thực tế khi có
# from model.knn.replay_model_knn import KNNModel

# Định nghĩa class cho một item trong request của /api/learn
class LearnItem(BaseModel):
    field_id: str
    ocr_value: str
    human_value: str
    suggestion_value: str

    class Config:
        json_schema_extra = {
            "example": {
                "field_id": "1",
                "ocr_value": "aple",
                "human_value": "apple",
                "suggestion_value": "apple"
            }
        }

# Định nghĩa class cho request body của /api/learn
class LearnRequest(BaseModel):
    data: List[LearnItem]

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"field_id": "1", "ocr_value": "aple", "human_value": "apple", "suggestion_value": ""},
                    {"field_id": "2", "ocr_value": "bannana", "human_value": "banana", "suggestion_value": ""}
                ]
            }
        }

# Định nghĩa class cho một item trong request của /api/suggest
class SuggestItem(BaseModel):
    field_id: str
    ocr_value: str

    class Config:
        json_schema_extra = {
            "example": {
                "field_id": "1",
                "ocr_value": "aple"
            }
        }

# Định nghĩa class cho request body của /api/suggest
class SuggestRequest(BaseModel):
    data: List[SuggestItem]

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"field_id": "1", "ocr_value": "aple"},
                    {"field_id": "2", "ocr_value": "bannana"}
                ]
            }
        }

# Định nghĩa class cho một item trong response của /api/suggest
class SuggestResponseItem(BaseModel):
    field_id: str
    ocr_value: str
    suggestion_value: str
    confidence: float

    class Config:
        json_schema_extra = {
            "example": {
                "field_id": "1",
                "ocr_value": "aple",
                "suggestion_value": "apple",
                "confidence": 0.95
            }
        }

# Định nghĩa class cho response body của /api/suggest
class SuggestResponse(BaseModel):
    data: List[SuggestResponseItem]

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"field_id": "1", "ocr_value": "aple", "suggestion_value": "apple", "confidence": 0.95},
                    {"field_id": "2", "ocr_value": "bannana", "suggestion_value": "banana", "confidence": 0.92}
                ]
            }
        }

# Hàm khởi tạo model theo tên
def initialize_model(model_name: str):
    model_map = {
        "tfidf": TfidfModel,
    }
    if model_name.lower() not in model_map:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(model_map.keys())}")
    return model_map[model_name.lower()]()