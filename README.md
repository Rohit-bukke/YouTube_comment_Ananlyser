# 📊 YouTube Comment Sentiment Analyzer & NLP Intelligence System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?logo=scikitlearn)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://docker.com)
[![CI](https://github.com/Rohit-bukke/YouTube_comment_Ananlyser/actions/workflows/ci.yml/badge.svg)](https://github.com/Rohit-bukke/YouTube_comment_Ananlyser/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Production-grade, end-to-end NLP system** for real-time sentiment classification of YouTube comments. Features multi-model benchmarking, a FastAPI REST microservice, an interactive Streamlit analytics dashboard, and Docker-ready deployment — engineered as a portfolio-ready ML project.

---

## Key Highlights

| Metric | Value |
|---|---|
| **Best Model** | Calibrated Linear SVC |
| **Macro F1-Score** | **84.13%** |
| **Accuracy** | **85.69%** |
| **Training Samples** | 28,988 |
| **Test Samples** | 7,248 |
| **Vocabulary Size** | 25,000 features |
| **Avg Inference Latency** | ~0.026 ms/sample |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                          │
│  Remote CSV / YouTube Data API v3 / Local Cache                 │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│               TEXT PREPROCESSING ENGINE                          │
│  URL Removal → Contraction Expansion → Sentiment-Aware          │
│  Stopword Filtering → WordNet Lemmatization                     │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│             FEATURE ENGINEERING (TF-IDF)                        │
│  Sublinear TF · Bigram N-grams · 25K Vocabulary Cap             │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│          MULTI-MODEL BENCHMARKING & SELECTION                   │
│  Naive Bayes │ Logistic Regression │ LinearSVC │ SGD │ Ensemble │
└──────────────────────┬───────────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌─────────────────┐      ┌──────────────────┐
│  FastAPI REST    │      │  Streamlit Web   │
│  Microservice    │      │  Dashboard       │
│  /predict        │      │  Live YouTube    │
│  /predict/batch  │      │  Analysis        │
│  /analyze/youtube│      │  Batch CSV       │
│  /health         │      │  Benchmarks      │
└─────────────────┘      └──────────────────┘
```

---

## Model Benchmarking Results

| Model Architecture | Accuracy | Macro F1 | Weighted F1 | Precision | Recall | Latency (ms) |
|---|---|---|---|---|---|---|
| **Calibrated LinearSVC** | **85.69%** | **84.13%** | **85.41%** | **85.52%** | **83.49%** | 0.026 |
| Logistic Regression | 84.41% | 83.31% | 84.32% | 83.48% | 83.53% | 0.003 |
| Soft Voting Ensemble | 83.94% | 82.29% | 83.62% | 83.93% | 81.48% | 0.038 |
| SGD Classifier | 79.62% | 77.58% | 79.16% | 79.70% | 77.13% | 0.006 |
| Multinomial Naive Bayes | 65.56% | 61.41% | 63.75% | 72.11% | 59.97% | 0.004 |

> Best model selected automatically via **Macro F1-Score** optimization across 3-class sentiment classification (Positive / Neutral / Negative).

---

## Project Structure

```
YouTube_comment_Ananlyser/
│
├── configs/
│   └── config.yaml                 # Centralized YAML configuration
│
├── src/
│   ├── data/
│   │   ├── loader.py               # Dataset ingestion, validation & caching
│   │   └── preprocessor.py         # Production NLP text cleaning pipeline
│   ├── features/
│   │   └── vectorizer.py           # Sublinear TF-IDF feature extractor
│   ├── models/
│   │   ├── trainer.py              # Multi-model benchmarking & training
│   │   ├── evaluate.py             # Metrics, confusion matrix & latency profiling
│   │   └── predictor.py            # Thread-safe inference engine
│   ├── services/
│   │   └── youtube_service.py      # YouTube comment extraction service
│   └── api/
│       ├── app.py                  # FastAPI REST microservice
│       └── schemas.py              # Pydantic request/response schemas
│
├── notebooks/
│   ├── 01_data_preprocessing_and_eda.ipynb
│   └── 02_model_experiments_and_benchmarking.ipynb
│
├── tests/
│   ├── test_preprocessor.py        # Text cleaning edge case tests
│   ├── test_vectorizer.py          # Feature extraction tests
│   ├── test_model.py               # Model evaluation & inference tests
│   └── test_api.py                 # API endpoint integration tests
│
├── app.py                          # Streamlit interactive dashboard
├── Dockerfile                      # Production container image
├── docker-compose.yml              # Multi-service orchestration
├── requirements.txt                # Dependency manifest
├── setup.py                        # Package configuration
└── README.md
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Rohit-bukke/YouTube_comment_Ananlyser.git
cd YouTube_comment_Ananlyser
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 2. Train the Model

```bash
python -m src.models.trainer
```

This will automatically download the dataset, preprocess text, benchmark 5 model architectures, select the best one, and export the production pipeline to `models/sentiment_pipeline.joblib`.

### 3. Launch Streamlit Dashboard

```bash
streamlit run app.py
```

### 4. Launch FastAPI Microservice

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

API documentation available at: `http://localhost:8000/docs`

### 5. Docker Deployment

```bash
docker-compose up --build
```

- **API**: `http://localhost:8000`
- **Dashboard**: `http://localhost:8501`

---

## API Reference

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This tutorial is absolutely brilliant!"}'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"comments": ["Amazing video!", "Terrible quality.", "What library is this?"]}'
```

### YouTube Video Analysis
```bash
curl -X POST http://localhost:8000/analyze/youtube \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://www.youtube.com/watch?v=VIDEO_ID", "max_comments": 50}'
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Resume Highlight Bullets

Use these impact-driven bullet points for your resume or LinkedIn:

- **Engineered a production-grade NLP sentiment classifier** achieving **84.13% Macro F1-Score** on 36K+ social media comments using Calibrated LinearSVC with sublinear TF-IDF (25K bigram features)
- **Benchmarked 5 ML architectures** (Naive Bayes, Logistic Regression, LinearSVC, SGD, Voting Ensemble) with automated best-model selection via cross-validated Macro F1 optimization
- **Built a real-time inference microservice** using FastAPI with sub-millisecond latency (0.026ms/sample), Pydantic schema validation, and batch processing endpoints
- **Developed an interactive Streamlit analytics dashboard** supporting live YouTube URL analysis, single-comment classification with confidence gauges, and batch CSV processing
- **Implemented end-to-end MLOps pipeline** with YAML-driven configuration, structured logging, CI/CD via GitHub Actions, Docker containerization, and comprehensive pytest test suite

---

## Tech Stack

| Category | Technologies |
|---|---|
| **ML / NLP** | scikit-learn, NLTK, TF-IDF, LinearSVC, CalibratedClassifierCV |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Streamlit, Matplotlib, Seaborn |
| **Data** | Pandas, NumPy, SciPy |
| **DevOps** | Docker, GitHub Actions CI/CD |
| **Testing** | Pytest |

---

## License

This project is open source under the [MIT License](LICENSE).
