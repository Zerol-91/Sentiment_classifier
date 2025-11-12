import os
import logging
import time
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
from dotenv import load_dotenv

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
        # Для Docker на Render используем внутренний адрес
        self.api_url = os.getenv('API_URL', 'http://localhost:10000')
        self.logger = logging.getLogger(__name__)
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        welcome_text = """
🤖 Sentiment Analyzer Bot (Render Edition)

Отправь текст на английском для анализа тональности!

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
• Я проанализирую его тональность
• Верну результат: позитивный/негативный и уверенность

Команды:
/start - начать работу
/help - эта справка
/status - статус бота
        """
        await update.message.reply_text(help_text)
    
    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /status"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                status_data = response.json()
                status_msg = f"✅ API статус: {status_data.get('status', 'unknown')}\n"
                status_msg += f"📊 Модель загружена: {status_data.get('model_loaded', False)}"
            else:
                status_msg = "❌ API недоступен"
        except Exception as e:
            status_msg = f"❌ Ошибка подключения к API: {str(e)}"
        
        await update.message.reply_text(status_msg)
    
    async def analyze_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений"""
        user_text = update.message.text
        user_name = update.message.from_user.first_name
        
        self.logger.info(f"Получен текст от {user_name}: {user_text[:50]}...")
        
        # Показываем что бот "печатает"
        await update.message.chat.send_action(action="typing")
        
        try:
            # Отправляем запрос к нашему API
            response = requests.post(
                f"{self.api_url}/predict",
                json={"text": user_text},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                sentiment = result['sentiment']
                confidence = result['confidence']
                processing_time = result['processing_time']
                
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
                
            else:
                reply_text = "❌ Ошибка при анализе текста. Попробуйте позже."
                
        except requests.exceptions.ConnectionError:
            reply_text = "🔌 API недоступен. Подождите немного..."
        except requests.exceptions.Timeout:
            reply_text = "⏰ Таймаут подключения к API. Попробуйте позже."
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка: {e}")
            reply_text = "⚠️ Произошла непредвиденная ошибка. Попробуйте еще раз."
        
        await update.message.reply_text(reply_text)
    
    def run(self):
        """Запуск бота с обработкой ошибок"""
        if not self.token:
            self.logger.error("❌ Токен бота не найден! Проверьте файл .env")
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
                    retry_delay *= 2  # Экспоненциальная задержка
                else:
                    self.logger.error("❌ Все попытки запуска бота провалились")

def main():
    """Основная функция для запуска бота отдельно"""
    bot = SentimentBot()
    bot.run()

if __name__ == '__main__':
    main()