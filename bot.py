import os
import logging
import time
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
import joblib
import numpy as np

# Настраиваем логирование
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения
load_dotenv()

class SentimentBot:
    def __init__(self):
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Загрузка ML модели"""
        try:
            model_path = os.getenv("MODEL_PATH", "models/baseline_model.pkl")
            self.logger.info(f"🤖 Загрузка модели из {model_path}...")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Модель не найдена: {model_path}")
                
            self.model = joblib.load(model_path)
            self.logger.info("✅ Модель успешно загружена!")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки модели: {e}")
            self.model = None
    
    def analyze_sentiment(self, text):
        """Анализ тональности текста с помощью модели"""
        if self.model is None:
            raise Exception("Модель не загружена")
        
        start_time = time.time()
        
        # Предсказание с помощью модели
        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        
        # Уверенность для позитивного класса
        confidence = float(probabilities[1] if prediction == "positive" else probabilities[0])
        processing_time = time.time() - start_time
        
        return prediction, confidence, processing_time
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        welcome_text = """
🤖 Sentiment Analyzer Bot (Standalone Edition)

Отправь текст на английском для анализа тональности!

Бот работает полностью самостоятельно с ML моделью 🧠

Команды:
/start - начать
/help - помощь  
/status - статус бота
        """
        await update.message.reply_text(welcome_text)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = """
📖 Помощь по боту:

• Просто отправь мне текст на английском
• Я проанализирую его тональность с помощью ML модели
• Верну результат: позитивный/негативный и уверенность

Команды:
/start - начать работу
/help - эта справка
/status - статус бота
        """
        await update.message.reply_text(help_text)
    
    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /status"""
        if self.model is not None:
            status_msg = "✅ Бот работает, модель загружена"
        else:
            status_msg = "❌ Модель не загружена"
        
        await update.message.reply_text(status_msg)
    
    async def analyze_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений"""
        user_text = update.message.text
        user_name = update.message.from_user.first_name
        
        self.logger.info(f"Получен текст от {user_name}: {user_text[:50]}...")
        
        # Показываем что бот "печатает"
        await update.message.chat.send_action(action="typing")
        
        try:
            # Анализируем текст напрямую с помощью модели
            sentiment, confidence, processing_time = self.analyze_sentiment(user_text)
            
            # Форматируем ответ
            if sentiment == 'positive':
                emoji = "😊"
                message = "ПОЗИТИВНЫЙ"
            else:
                emoji = "😠" 
                message = "НЕГАТИВНЫЙ"
            
            reply_text = f"""
{emoji} Результат анализа: {message}

📊 Уверенность: {confidence:.1%}
⏱ Время обработки: {processing_time:.3f} сек

Текст: "{user_text[:100]}{'...' if len(user_text) > 100 else ''}"
            """
                
        except Exception as e:
            self.logger.error(f"Ошибка анализа: {e}")
            reply_text = "❌ Ошибка при анализе текста. Попробуйте позже."
        
        await update.message.reply_text(reply_text)
    
    def run(self):
        """Запуск бота"""
        if not self.token:
            self.logger.error("❌ Токен бота не найден! Проверьте файл .env")
            return
        
        if self.model is None:
            self.logger.error("❌ Модель не загружена! Бот не может работать.")
            return
        
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                application = Application.builder().token(self.token).build()
                
                # Регистрируем обработчики команд
                application.add_handler(CommandHandler("start", self.start))
                application.add_handler(CommandHandler("help", self.help_command))
                application.add_handler(CommandHandler("status", self.status))
                
                # Регистрируем обработчик текстовых сообщений
                application.add_handler(MessageHandler(
                    filters.TEXT & ~filters.COMMAND, 
                    self.analyze_text
                ))
                
                self.logger.info("🤖 Бот запускается...")
                application.run_polling()
                break
                
            except Exception as e:
                self.logger.error(f"❌ Попытка {attempt + 1} не удалась: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"🔄 Повтор через {retry_delay} сек...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    self.logger.error("❌ Все попытки запуска бота провалились")

def main():
    """Основная функция"""
    bot = SentimentBot()
    bot.run()

if __name__ == '__main__':
    main()