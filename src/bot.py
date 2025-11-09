import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
from dotenv import load_dotenv

# Настраиваем логирование чтобы видеть что происходит
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения из .env файла
load_dotenv()

# Получаем токен бота из переменных окружения
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
API_URL = os.getenv('API_URL', 'http://localhost:8000')

class SentimentBot:
    def __init__(self):
        self.token = TELEGRAM_TOKEN
        self.api_url = API_URL
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        welcome_text = """
🤖 Добро пожаловать в Sentiment Analyzer Bot!

Отправь мне любой текст на английском, и я определю его тональность: позитивный 😊 или негативный 😠.

Примеры для тестирования:
• "I love this movie! Amazing acting!"
• "This is terrible, worst product ever."
• "It's okay, nothing special."

Просто напиши мне что-нибудь!
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
/stats - статистика бота
        """
        await update.message.reply_text(help_text)
    
    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stats"""
        stats_text = """
📊 Статистика бота:
• Модель: TF-IDF + Logistic Regression
• Точность: 87.5% (F1-score)
• Классы: positive/negative
• API: {api_url}
        """.format(api_url=self.api_url)
        await update.message.reply_text(stats_text)
    
    async def analyze_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений"""
        user_text = update.message.text
        user_name = update.message.from_user.first_name
        
        logger.info(f"Получен текст от {user_name}: {user_text[:50]}...")
        
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
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка подключения к API: {e}")
            reply_text = "🔌 Не могу подключиться к сервису анализа. Убедитесь что API сервер запущен."
        
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            reply_text = "⚠️ Произошла непредвиденная ошибка. Попробуйте еще раз."
        
        await update.message.reply_text(reply_text)
    
    def run(self):
        """Запуск бота"""
        if not self.token:
            logger.error("Токен бота не найден! Проверь файл .env")
            return
        
        # Создаем приложение бота
        application = Application.builder().token(self.token).build()
        
        # Регистрируем обработчики команд
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("stats", self.stats))
        
        # Регистрируем обработчик текстовых сообщений
        application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, 
            self.analyze_text
        ))
        
        # Запускаем бота
        logger.info("Бот запускается...")
        application.run_polling()

def main():
    """Основная функция"""
    bot = SentimentBot()
    bot.run()

if __name__ == '__main__':
    main()