import logging
from fastapi import FastAPI, HTTPException
from common import LearnRequest, SuggestRequest, SuggestResponse, initialize_model
import uvicorn

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OCR Text Correction API",
    description="API for suggesting and learning OCR text corrections using multiple models (e.g., TF-IDF, KNN).",
    version="1.0.1",
    openapi_tags=[
        {
            "name": "tfidf",
            "description": "Endpoints for TF-IDF based text correction model."
        }
       
    ]
)
DEFAULT_MODEL = "tfidf"  # Mặc định sử dụng model TF-IDF

# Cache các model để tránh khởi tạo lại
models = {
    DEFAULT_MODEL: initialize_model(DEFAULT_MODEL),    
}

@app.get("/")
def read_root():
    return {"Service status": "Running"}

@app.post(
    "/api/{model_name}/learn",
    tags=["tfidf"],
    summary="Train the specified model with correction data",
    description="Receives a list of correction data to train the specified model (e.g., TF-IDF, KNN). Each item contains field_id, OCR text, human-corrected text, and suggested correction."
)
async def learn(model_name: str, request: LearnRequest):
    try:
        # Khởi tạo hoặc lấy model từ cache
        if model_name.lower() not in models:
            models[model_name.lower()] = initialize_model(model_name)
        model = models[model_name.lower()]
        
        # Gọi phương thức learn cho từng item
        for item in request.data:
            model.learn(
                field=item.field_id,
                ocr_value=item.ocr_value,
                human_value=item.human_value,
                suggestion=item.suggestion_value
            )
        return {"status": "success", "message": f"{model_name.upper()} model updated successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during learning: {str(e)}")

@app.post(
    "/api/{model_name}/suggest",
    response_model=SuggestResponse,
    tags=["tfidf"],
    summary="Generate correction suggestions using the specified model",
    description="Receives a list of OCR text data and returns suggested corrections with confidence scores using the specified model (e.g., TF-IDF, KNN)."
)
async def suggest(model_name: str, request: SuggestRequest):
    try:
        # Khởi tạo hoặc lấy model từ cache
        if model_name.lower() not in models:
            models[model_name.lower()] = initialize_model(model_name)
        model = models[model_name.lower()]
        
        # Gọi phương thức suggest cho từng item và định dạng response
        response_data = []
        for item in request.data:
            suggestion, confidence = model.suggest(
                field=item.field_id,
                ocr_value=item.ocr_value
            )
            # Đảm bảo suggestion_value là chuỗi, sử dụng ocr_value làm fallback nếu suggestion là None
            suggestion_value = suggestion if suggestion is not None else ""
            response_data.append({
                "field_id": item.field_id,
                "ocr_value": item.ocr_value,
                "suggestion_value": suggestion_value,
                "confidence": float(confidence)  # Đảm bảo confidence là float
            })
            logger.debug(f"Done suggestion by model {model_name} for field '{item.field_id}' with OCR value '{item.ocr_value}': {suggestion_value} - confidence: {confidence}")
        return {"data": response_data}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during suggestion: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting API server...")
    # Uncomment the line below to run the server directly or debug it
    #uvicorn.run(app, host="0.0.0.0", port=8000)