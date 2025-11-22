"""
Скрипт для обучения baseline модели
"""
import os
import sys

sys.path.append('src')

from preprocess import load_sentiment_data
from model import BaselineModel

def main():
    
    os.makedirs('models', exist_ok=True)
    
    print("Начинаем обучение baseline модели...")
    
    print("Этап 1: Загрузка данных...")
    train_df, test_df = load_sentiment_data()
    
    print("Этап 2: Обучение модели...")
    model = BaselineModel()
    model.train(train_df['cleaned_text'], train_df['sentiment'])
    
    print("Этап 3: Оценка модели...")
    f1_score = model.evaluate(test_df['cleaned_text'], test_df['sentiment'])
    
    print("Этап 4: Сохранение модели...")
    model.save()
    
    if f1_score >= 0.75:
        print(f"F1-score: {f1_score:.3f}")
    
    return f1_score

if __name__ == "__main__":
    main()