from datasets import load_dataset 
import pandas as pd  
import re  



def load_sentiment_data():

    dataset = load_dataset("imdb")
    
    train_df = pd.DataFrame({
        'text': dataset['train']['text'],
        'label': dataset['train']['label']
    })
    
    test_df = pd.DataFrame({
        'text': dataset['test']['text'],
        'label': dataset['test']['label']
    })
    
    print(f"Загружено {len(train_df)} тренировочных и {len(test_df)} тестовых отзывов")
    
    # ДОБАВЛЯЕМ ОЧИСТКУ ТЕКСТА ПРЯМО ЗДЕСЬ!
    print("Очищаем текст...")
    train_df['cleaned_text'] = train_df['text'].apply(clean_text)
    test_df['cleaned_text'] = test_df['text'].apply(clean_text)
    
    # Преобразуем метки в читаемый вид
    train_df['sentiment'] = train_df['label'].map({0: 'negative', 1: 'positive'})
    test_df['sentiment'] = test_df['label'].map({0: 'negative', 1: 'positive'})
    
    return train_df, test_df


def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = text.lower()

    text = ' '.join(text.split())
    return text


if __name__ == "__main__":
    print("Запускаем подготовку данных...")
    
    # Вызываем нашу функцию
    train_data, test_data = load_sentiment_data()
    

    print("Статистика данных:")
    print(f"Тренировочные данные: {train_data['sentiment'].value_counts()}")
    print(f"Тестовые данные: {test_data['sentiment'].value_counts()}")
    
    print("\nПримеры данных:")
    for i in range(2):  # Покажем 2 примера
        print(f"Текст: {train_data['text'].iloc[i][:100]}...")  # первые 100 символов
        print(f"Очищенный: {train_data['cleaned_text'].iloc[i][:100]}...")
        print(f"Настроение: {train_data['sentiment'].iloc[i]}")
        print("---")