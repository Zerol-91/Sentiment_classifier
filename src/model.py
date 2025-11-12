from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
import joblib

class BaselineModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                stop_words='english', 
                ngram_range=(1, 2)
            )),
            ('clf', LogisticRegression(
                random_state=42,
                max_iter=1000
            ))
        ])
    
    def train(self, texts, labels):
        print("🎯 Обучаем baseline модель...")
        self.pipeline.fit(texts, labels)
        print("✅ Модель обучена!")
    
    def predict(self, texts):
        return self.pipeline.predict(texts)
    
    def predict_proba(self, texts):
        return self.pipeline.predict_proba(texts)
    
    def evaluate(self, test_texts, test_labels):
        predictions = self.predict(test_texts)
        
        print("📊 Результаты оценки модели:")
        print(classification_report(test_labels, predictions))
        
        f1 = f1_score(test_labels, predictions, average='macro')
        print(f"🎯 Macro F1 Score: {f1:.3f}")
        
        return f1
    
    def save(self, path='models/baseline_model.pkl'):
        joblib.dump(self.pipeline, path)
        print(f"💾 Модель сохранена в {path}")
    
    def load(self, path='models/baseline_model.pkl'):
        """Загрузка модели из файла"""
        self.pipeline = joblib.load(path)
        print(f"📥 Модель загружена из {path}")
