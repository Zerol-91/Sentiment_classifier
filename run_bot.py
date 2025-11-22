"""
Главный файл запуска Telegram бота
"""

import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

load_dotenv()

def main():
    """Запускаем бота с проверкой всех зависимостей"""
        
    required_vars = ['TELEGRAM_BOT_TOKEN']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Ошибка: отсутствуют переменные окружения: {', '.join(missing_vars)}")
        print("Создайте файл .env и добавьте:")
        for var in missing_vars:
            print(f"   {var}=your_value_here")
        return
    
    model_path = os.getenv('MODEL_PATH', 'models/baseline_model.pkl')
    if not os.path.exists(model_path):
        print(f"Ошибка: файл модели не найден: {model_path}")
        print("Выполните: python train_baseline.py")
        return
    
    print("Запускаем Telegram бота...")
    print(f"Модель: {model_path}")
    
    try:
        from src.bot import main as bot_main
        bot_main()
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Проверьте структуру проекта и зависимости")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")

if __name__ == '__main__':
    main()