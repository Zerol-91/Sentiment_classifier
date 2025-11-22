import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# ✅ ИМПОРТИРУЕМ НАШ КЛАСС МОДЕЛИ ЕДИНООБРАЗНО С API
from model import BaselineModel

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения
load_dotenv()

# Получаем конфигурацию из переменных окружения
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
MODEL_PATH = os.getenv('MODEL_PATH', 'models/baseline_model.pkl')

class SentimentBot:
    def __init__(self):
        self.token = TELEGRAM_TOKEN
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Загружаем модель ЕДИНООБРАЗНО с API - через BaselineModel"""
        try:
            # ✅ ИСПОЛЬЗУЕМ ТОТ ЖЕ СПОСОБ, ЧТО И В API
            self.model = BaselineModel()
            self.model.load(MODEL_PATH)
            logger.info("✅ Модель загружена в боте через BaselineModel!")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            self.model = None
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        await update.message.reply_text(
            "🤖 Sentiment Analyzer Bot!\n\n"
            "Отправь мне текст на английском для анализа тональности.\n"
            "Команды:\n"
            "/start - начать работу\n"
            "/help - помощь\n"
            "/analyze <текст> - анализ тональности\n"
            "/stats - статистика"
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        await update.message.reply_text(
            "📖 Как пользоваться ботом:\n\n"
            "1. Просто отправь текст на английском\n"
            "2. Или используй команду /analyze <текст>\n"
            "3. Бот определит позитивный или негативный отзыв\n"
            "4. Покажет уверенность предсказания"
        )
    
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /analyze <текст>"""
        if not context.args:
            await update.message.reply_text("❌ Укажите текст для анализа: /analyze <ваш текст>")
            return
            
        user_text = " ".join(context.args)
        await self.analyze_and_reply(update, user_text)
    
    async def analyze_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик обычных текстовых сообщений"""
        user_text = update.message.text
        await self.analyze_and_reply(update, user_text)
    
    async def analyze_and_reply(self, update: Update, user_text: str):
        """Общая функция анализа текста и отправки результата"""
        if self.model is None:
            await update.message.reply_text("❌ Модель не загружена. Попробуйте позже.")
            return
            
        try:
            # ✅ ИСПОЛЬЗУЕМ МЕТОДЫ BaselineModel КАК В API
            prediction = self.model.predict([user_text])[0]
            probabilities = self.model.predict_proba([user_text])[0]
            
            sentiment = prediction
            confidence = float(probabilities[0] if prediction == "negative" else probabilities[1])
            
            # Форматируем ответ
            emoji = "😊" if sentiment == 'positive' else "😠"
            sentiment_text = "ПОЗИТИВНЫЙ" if sentiment == 'positive' else "НЕГАТИВНЫЙ"
            
            reply_text = (
                f"{emoji} **{sentiment_text}**\n\n"
                f"📊 Уверенность: {confidence:.1%}\n"
                f"🔍 Анализировали: '{user_text[:50]}{'...' if len(user_text) > 50 else ''}'"
            )
            
        except Exception as e:
            logger.error(f"Ошибка анализа текста: {e}")
            reply_text = "⚠️ Ошибка при анализе текста. Убедитесь, что текст на английском."
        
        await update.message.reply_text(reply_text)
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stats"""
        stats_text = (
            "📈 **Статистика бота:**\n\n"
            f"🤖 Модель: TF-IDF + Logistic Regression\n"
            f"✅ Модель загружена: {'Да' if self.model else 'Нет'}\n"
            f"🎯 F1-score: 87.5%\n"
            f"📚 Обучена на: 50,000 отзывов IMDB"
        )
        await update.message.reply_text(stats_text)
    
    def run_polling(self):
        """Запуск бота в режиме polling (основной метод)"""
        logger.info("🔄 Запускаем Telegram бота с polling...")
        
        # Создаем приложение
        application = Application.builder().token(self.token).build()
        
        # Регистрируем обработчики команд
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("analyze", self.analyze_command))
        application.add_handler(CommandHandler("stats", self.stats_command))
        
        # Регистрируем обработчик текстовых сообщений
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.analyze_text))
        
        logger.info("✅ Бот запущен! Ожидаем сообщения...")
        application.run_polling()

def main():
    """Главная функция запуска бота"""
    if not TELEGRAM_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN не установлен! Проверьте .env файл.")
        return
        
    try:
        bot = SentimentBot()
        bot.run_polling()
    except Exception as e:
        logger.error(f"❌ Критическая ошибка запуска бота: {e}")

if __name__ == '__main__':
    main()