from pydantic import BaseModel
from typing import List
from model.tfidf.tfidf import TfidfModel
# Giả định có KNNModel, thay thế bằng import thực tế khi có
# from model.knn.replay_model_knn import KNNModel

# Định nghĩa class cho một item trong request của /api/learn
class LearnItem(BaseModel):
    project_id: str = None  # Dùng None nếu không cần thiết
    project_name: str = None  # Dùng None nếu không cần thiết
    template_id: str = None  # Dùng None nếu không cần thiết
    template_name: str = None  # Dùng None nếu không cần thiết
    field_id: str
    field_name: str = None  # Dùng None nếu không cần thiết
    ocr_value: str
    human_value: str
    suggestion_value: str

    class Config:
        json_schema_extra = {
            "example": {
                "project_id": "123",
                "project_name": "Test Project",
                "template_id": "456",
                "template_name": "Giấy khai sinh",
                "field_id": "1",
                "field_name": "Nơi sinh",
                "ocr_value": "Ha Nội",
                "human_value": "Hà Nội",
                "suggestion_value": "Hà Nội"
            }
        }

# Định nghĩa class cho request body của /api/learn
class LearnRequest(BaseModel):
    data: List[LearnItem]

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {   
                        "project_id": "123",
                        "project_name": "Test Project",
                        "template_id": "456",
                        "template_name": "Giấy khai sinh",
                        "field_id": "1",
                        "field_name": "Nơi sinh",
                        "ocr_value": "Ha Nội",
                        "human_value": "Hà Nội",    
                        "suggestion_value": "Hà Nội"
                    },
                    {
                        "project_id": "123",
                        "project_name": "Test Project",
                        "template_id": "456",
                        "template_name": "Giấy khai sinh",
                        "field_id": "2",
                        "field_name": "Quốc tịch",
                        "ocr_value": "Viet Nam",
                        "human_value": "Việt Nam",
                        "suggestion_value": "Việt Nam"
                    }
                    
                ]
            }
        }

# Định nghĩa class cho một item trong request của /api/suggest
class SuggestItem(BaseModel):
    project_id: str = None  # Dùng None nếu không cần thiết
    project_name: str = None  # Dùng None nếu không cần thiết
    template_id: str = None  # Dùng None nếu không cần thiết
    template_name: str = None  # Dùng None nếu không cần thiết
    field_id: str
    field_name: str = None  # Dùng None nếu không cần thiết
    ocr_value: str

    class Config:
        json_schema_extra = {
            "example":{
                "project_id": "123",
                "project_name": "Test Project", 
                "template_id": "456",
                "template_name": "Giấy khai sinh",
                "field_id": "1",
                "field_name": "Nơi sinh",
                "ocr_value": "Ha Noi"
            }
        }

# Định nghĩa class cho request body của /api/suggest
class SuggestRequest(BaseModel):
    data: List[SuggestItem]

    class Config:
        json_schema_extra = {
            "example": {
                "data": 
                    [{
                        "project_id": "123",
                        "project_name": "Test Project", 
                        "template_id": "456",
                        "template_name": "Giấy khai sinh",  
                        "field_id": "1",
                        "field_name": "Nơi sinh",
                        "ocr_value": "Ha Noi"
                    }
                ]
            }
        }

# Định nghĩa class cho một item trong response của /api/suggest
class SuggestResponseItem(BaseModel):
    project_id: str = None  # Dùng None nếu không cần thiết
    project_name: str = None  # Dùng None nếu không cần thiết
    template_id: str = None  # Dùng None nếu không cần thiết
    template_name: str = None  # Dùng None nếu không cần thiết
    field_id: str
    field_name: str = None  # Dùng None nếu không cần thiết
    ocr_value: str
    suggestion_value: str
    confidence: float

    class Config:
        json_schema_extra = {
            "example": {
                "project_id": "123",
                "project_name": "Test Project",
                "template_id": "456",
                "template_name": "Giấy khai sinh",
                "field_id": "1",
                "field_name": "Nơi sinh",
                "ocr_value": "Ha Noi",
                "suggestion_value": "Hà Nội",    
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
                   {
                       "project_id": "123",
                       "project_name": "Test Project",
                       "template_id": "456",
                       "template_name": "Giấy khai sinh",
                       "field_id": "1",
                       "field_name": "Nơi sinh",
                       "ocr_value": "Ha Noi",
                       "suggestion_value": "Hà Nội",
                       "confidence": 0.95
                   }
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