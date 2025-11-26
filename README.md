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

### Option 1: Live Demo Bot (Fastest)
Try the pre-deployed bot - no installation required!

1. Open [Telegram](https://telegram.org)
2. Search for `@Sentiment91_bot` 
3. Send `/start` command
4. Type any English text for sentiment analysis

If the bot doesn't respond, then you need to launch the container yourself via the zerol91/sentiment-classifier image. You can find it on the Docker Hub.

### Option 2: Docker with Your Own Bot
Run the pre-built Docker image with your own Telegram bot:

```bash
# Pull the Docker image
docker pull zerol91/sentiment-classifier:latest

# Create your bot with @BotFather and get token
# Run container with your token
docker run -d \
  --name sentiment-bot \
  -e TELEGRAM_BOT_TOKEN="your_bot_token_here" \
  zerol91/sentiment-classifier:latest
  ```

### Option 3: Local Installation
For development and customization:

```bash
# Clone repository
git clone https://github.com/zerol91/sentiment-classifier.git
cd sentiment-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your Telegram bot token

# Run the bot
python run_bot.py
```

### 🔧 Configuration
Create a .env file with the following variables:
```
# .env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
MODEL_PATH=models/baseline_model.pkl
API_HOST=localhost
API_PORT=8000
```

## Telegram Bot
Commands
```
/start - Welcome message and instructions

/help - Usage guide

/analyze <text> - Analyze specific text ("/analyze" is optional)

/stats - Model information and performance
```


## Example Usage Rest API (Optional)
The sentiment analysis model can also be accessed via REST API:

### Start API Server
```bash
python run_api.py
```

### API Documentation
Once running, visit: http://localhost:8000/docs
Endpoints
```
GET / - API status
GET /health - Health check and model status
POST /predict - Analyze text sentiment
GET /docs - Interactive API documentation
```

The FastAPI interactive docs make it easy to test the sentiment analysis:

Step-by-Step Guide:
- On the docs page, find the POST /predict endpoint in the "default" section
- Look for the green "POST" button with the /predict path

- Click the "Try it out" button to enable interactive testing

- Enter Your Text
In the request body, replace the example with your own text:
```json
{
  "text": "This movie is absolutely amazing and I love it!"
}
```
- Click the "Execute" button to send the request

Example Responce:
```json
{
  "sentiment": "positive",
  "confidence": 0.7164391312416224,
  "processing_time": 0.004557132720947266,
  "model_version": "baseline_v1"
}
```

## Docker Development
Build Image
```bash
docker build -t sentiment-classifier .
```

Run with Docker Compose
```bash
docker-compose up -d
```

View Logs
```bash
docker logs -f sentiment-bot
```

Testing
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

