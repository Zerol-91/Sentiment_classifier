#!/usr/bin/env python3
"""
Главный файл для запуска на Render через Docker
Объединяет API и бота в одном контейнере
"""
import os
import threading
import logging
import time
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_api():
    """Запуск API сервера"""
    try:
        from api import app
        import uvicorn
        
        port = int(os.getenv("PORT", 10000))
        logger.info(f"🚀 Запуск API на порту {port}")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        logger.error(f"❌ Ошибка запуска API: {e}")
        raise

def run_bot():
    """Запуск Telegram бота в отдельном потоке"""
    try:
        # Ждем немного пока API поднимется
        time.sleep(3)
        
        from bot import SentimentBot
        logger.info("🤖 Запуск Telegram бота...")
        bot = SentimentBot()
        bot.run()
    except Exception as e:
        logger.error(f"❌ Ошибка запуска бота: {e}")

def check_environment():
    """Проверка обязательных переменных окружения"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("❌ TELEGRAM_BOT_TOKEN не установлен!")
        return False
    
    # Проверяем наличие модели
    model_path = os.getenv("MODEL_PATH", "models/baseline_model.pkl")
    if not os.path.exists(model_path):
        logger.warning(f"⚠️ Модель не найдена: {model_path}")
        # Не падаем, т.к. модель может загрузиться позже
    
    return True

def main():
    """Главная функция запуска"""
    logger.info("Запуск объединенного сервиса API + Bot в Docker")
    
    if not check_environment():
        return
    
    # Запускаем бота в отдельном потоке (daemon=True)
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    logger.info("✅ Бот запущен в фоновом потоке")
    
    # Запускаем API в основном потоке (блокирующий вызов)
    run_api()

if __name__ == "__main__":
    main()