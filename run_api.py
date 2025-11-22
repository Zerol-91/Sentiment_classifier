
import uvicorn
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

if __name__ == "__main__":
    # Запускаем сервер с помощью uvicorn
    uvicorn.run(
        "src.api:app",          
        host="0.0.0.0",         
        port=8000,              
        reload=True, # Автоматическая перезагрузка при изменении кода (только для разработки) - в продакшене False
        log_level="info"        
    )