"""
Скрипт для обучения baseline модели
"""
import os
import sys
# Добавляем src в путь для импорта
sys.path.append('src')

from preprocess import load_sentiment_data
from model import BaselineModel

def main():
    # Создаем папку для моделей
    os.makedirs('models', exist_ok=True)
    
    print("🚀 Начинаем обучение baseline модели...")
    
    # 1. Загружаем данные
    print("📥 Этап 1: Загрузка данных...")
    train_df, test_df = load_sentiment_data()
    
    # 2. Обучаем модель
    print("🎯 Этап 2: Обучение модели...")
    model = BaselineModel()
    model.train(train_df['cleaned_text'], train_df['sentiment'])
    
    # 3. Оцениваем модель
    print("📊 Этап 3: Оценка модели...")
    f1_score = model.evaluate(test_df['cleaned_text'], test_df['sentiment'])
    
    # 4. Сохраняем модель
    print("💾 Этап 4: Сохранение модели...")
    model.save()
    
    # Проверяем цель F1 >= 75%
    if f1_score >= 0.75:
        print(f"🎉 Цель достигнута! F1-score: {f1_score:.3f}")
    else:
        print(f"⚠️ Цель не достигнута. F1-score: {f1_score:.3f}")
    
    return f1_score

if __name__ == "__main__":
    main()