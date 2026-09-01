FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Copy project
COPY . .

# Train model if not already present
RUN python -c "import os; \
    exec('if not os.path.exists(\"models/sentiment_pipeline.joblib\"):\n    from src.models.trainer import ModelTrainer\n    ModelTrainer().train_and_benchmark()')"

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
