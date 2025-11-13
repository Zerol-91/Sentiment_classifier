FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Копируем модель в контейнер
COPY models/ ./models/

RUN mkdir -p models
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Запускаем бота
CMD ["python", "bot.py"]