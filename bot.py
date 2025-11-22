import os
import logging
import joblib
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
MODEL_PATH = os.getenv('MODEL_PATH', 'models/baseline_model.pkl')

class SentimentBot:
    def __init__(self):
        self.token = TELEGRAM_TOKEN
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Загружаем модель при инициализации"""
        try:
            self.model = joblib.load(MODEL_PATH)
            logger.info("✅ Модель загружена в боте!")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            self.model = None
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 Sentiment Analyzer Bot!\n\n"
            "Отправь мне текст на английском для анализа тональности."
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Просто отправь текст на английском")
    
    async def analyze_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.model is None:
            await update.message.reply_text("❌ Модель не загружена")
            return
            
        user_text = update.message.text
        
        try:
            # Используем модель напрямую
            prediction = self.model.predict([user_text])[0]
            probabilities = self.model.predict_proba([user_text])[0]
            
            sentiment = prediction
            confidence = float(probabilities[0] if prediction == "negative" else probabilities[1])
            
            emoji = "😊" if sentiment == 'positive' else "😠"
            message = "ПОЗИТИВНЫЙ" if sentiment == 'positive' else "НЕГАТИВНЫЙ"
            
            reply_text = f"{emoji} {message}\nУверенность: {confidence:.1%}"
            
        except Exception as e:
            logger.error(f"Ошибка анализа: {e}")
            reply_text = "⚠️ Ошибка при анализе текста."
        
        await update.message.reply_text(reply_text)
    
    def run_webhook(self):
        """Запуск бота с webhook для Render"""
        application = Application.builder().token(self.token).build()
        
        # Регистрируем обработчики
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.analyze_text))
        
        # Получаем URL из переменных Render
        render_url = os.getenv('RENDER_EXTERNAL_URL')
        if not render_url:
            logger.error("RENDER_EXTERNAL_URL не установлен! Используем polling.")
            return self.run_polling()
        
        # Настраиваем webhook
        webhook_url = f"{render_url}/webhook"
        logger.info(f"🔄 Устанавливаем webhook: {webhook_url}")
        
        application.run_webhook(
            listen="0.0.0.0",
            port=8000,
            url_path=self.token,
            webhook_url=webhook_url,
            secret_token='WEBHOOK_SECRET'
        )
    
    def run_polling(self):
        """Запуск бота с polling (для разработки)"""
        logger.info("🔄 Запускаем бота с polling...")
        application = Application.builder().token(self.token).build()
        
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.analyze_text))
        
        application.run_polling()

def main():
    if not TELEGRAM_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN не установлен!")
        return
        
    bot = SentimentBot()
    
    # Определяем режим запуска
    if os.getenv('RENDER'):
        # На Render используем webhook
        bot.run_webhook()
    else:
        # Локально используем polling
        bot.run_polling()

if __name__ == '__main__':
    main()