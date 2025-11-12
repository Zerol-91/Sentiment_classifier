FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Для Render - проверяем модель и запускаем объединенный сервис
CMD ["python", "main.py"]