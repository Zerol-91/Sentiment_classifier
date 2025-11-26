# Sentiment Analysis Classifier

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://docker.com)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-blue.svg)](https://telegram.org)

A production-ready machine learning system for sentiment analysis of English text. This project demonstrates a complete ML pipeline from data processing to deployment with a REST API and Telegram bot.


## 🚀 Features

- **ML Model**: TF-IDF + Logistic Regression with 87.5% F1-score
- **REST API**: FastAPI with automatic documentation and validation
- **Telegram Bot**: Real-time sentiment analysis in chat
- **Dockerized**: Ready for deployment with containerization
- **Production Ready**: Error handling, logging, and health checks

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **F1-Score** | 87.5% |
| **Accuracy** | 85.2% |
| **Precision** | 86.1% |
| **Recall** | 88.9% |

*Trained on 50,000 IMDB movie reviews*

## 🛠 Tech Stack

**Backend & ML**
- Python 3.9+
- FastAPI (REST API)
- Scikit-learn (ML model)
- Pandas & Numpy (Data processing)
- Joblib (Model serialization)

**Deployment & Infrastructure**
- Docker & Docker Compose
- Uvicorn (ASGI server)
- Python-telegram-bot

**Data**
- Hugging Face Datasets (IMDB reviews)
- NLTK (Text preprocessing)

## 🏗 Project Structure


```
sentiment-classifier/
├── src/
│ ├── api.py # FastAPI application
│ ├── bot.py # Telegram bot
│ ├── model.py # ML model class
│ └── preprocess.py # Data preprocessing
├── models/
│ └── baseline_model.pkl # Trained model
├── tests/ # Test suite
├── Dockerfile # Container configuration
├── docker-compose.yml # Multi-container setup
├── requirements.txt # Python dependencies
└── run_*.py # Application entry points
```


## ⚡ Quick Start

### Option 1: Docker (Recommended)

```bash
# Pull the pre-built image
docker pull zerol91/sentiment-classifier:latest

# Run the container
docker run -d \
  --name sentiment-bot \
  -e TELEGRAM_BOT_TOKEN="your_bot_token_here" \
  yourusername/sentiment-classifier:latest
```

```bash
# Clone the repository
git clone https://github.com/yourusername/sentiment-classifier.git
cd sentiment-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Telegram bot token

# Run the application
python run_bot.py
```

## 🔧 Configuration
Create a .env file with the following variables:
```
env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
MODEL_PATH=models/baseline_model.pkl
API_HOST=localhost
API_PORT=8000
```

## 📡 API Documentation
Endpoints
```
GET / - API status
GET /health - Health check and model status
POST /predict - Analyze text sentiment
GET /docs - Interactive API documentation
```

Example Usage
```bash
# Analyze sentiment
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is absolutely fantastic!"}'
# Response:

json
{
  "sentiment": "positive",
  "confidence": 0.92,
  "processing_time": 0.045,
  "model_version": "baseline_v1"
}
```
## Telegram Bot
Commands
```
/start - Welcome message and instructions

/help - Usage guide

/analyze <text> - Analyze specific text

/stats - Model information and performance
```

## Setup
Create a bot with BotFather on Telegram
Get your bot token
Set TELEGRAM_BOT_TOKEN in your environment (.env file)
Start chatting with your bot!


## Docker Development
Build Image
```bash
docker build -t sentiment-classifier .
Run with Docker Compose
```
```bash
docker-compose up -d
View Logs
```
```bash
docker logs -f sentiment-bot
```
🧪 Testing
```bash
# Run tests
pytest tests/

# Test API locally
python run_api.py

# Test bot locally  
python run_bot.py
```
## 📈 Model Training
To retrain the model with different parameters:

```bash
python train_baseline.py
```
The training process:
Loads and preprocesses IMDB data
Trains TF-IDF + Logistic Regression model
Evaluates performance
Saves model to models/baseline_model.pkl

