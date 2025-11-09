from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import os
from typing import List
import time

# Импортируем нашу модель
from model import BaselineModel

# Функция для загрузки модели при запуске
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - загружаем модель
    global model
    try:
        model = BaselineModel()
        model_path = "models/baseline_model.pkl"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
            
        model.load(model_path)
        print("✅ Модель успешно загружена!")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        model = None
    yield
    # Shutdown - можно добавить очистку ресурсов

# Создаем FastAPI приложение с lifespan
app = FastAPI(
    title="Sentiment Classifier API",
    description="API для классификации тональности текста",
    version="1.0.0",
    lifespan=lifespan
)

# Модели данных для запросов и ответов
class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    processing_time: float
    model_version: str = "baseline_v1"

class BatchPredictionResponse(BaseModel):
    predictions: List[dict]
    total_processed: int
    total_time: float

# Эндпоинты API
@app.get("/")
async def root():
    return {
        "message": "Sentiment Classifier API", 
        "status": "active",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: TextRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    start_time = time.time()
    
    try:
        # Предсказание
        prediction = model.predict([request.text])[0]
        probabilities = model.predict_proba([request.text])[0]
        
        # Определяем уверенность предсказания
        confidence = float(probabilities[0] if prediction == "negative" else probabilities[1])
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            sentiment=prediction,
            confidence=confidence,
            processing_time=processing_time,
            model_version="baseline_v1"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")

@app.get("/model_info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    return {
        "model_type": "TF-IDF + Logistic Regression",
        "version": "baseline_v1",
        "classes": ["negative", "positive"],
        "f1_score": 0.875
    }

# Запуск сервера для разработки
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)