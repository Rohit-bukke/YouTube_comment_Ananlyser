"""
Production FastAPI Application for Sentiment Analysis & YouTube Intelligence.
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BatchCommentRequest,
    BatchPredictionResponse,
    CommentRequest,
    HealthResponse,
    PredictionResponse,
    YouTubeAnalysisRequest,
    YouTubeAnalysisResponse,
    YouTubeCommentResult,
)
from src.models.predictor import SentimentPredictor
from src.services.youtube_service import YouTubeService
from src.utils.config_manager import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

# Global predictor & service instances
predictor: SentimentPredictor = None
youtube_service: YouTubeService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and pre-warms ML models and services on startup."""
    global predictor, youtube_service
    logger.info("Initializing Sentiment Analysis microservice...")
    predictor = SentimentPredictor(config=config)
    youtube_service = YouTubeService()
    if predictor.is_ready:
        logger.info("Model pipeline pre-warmed and ready.")
    else:
        logger.warning("Model pipeline not yet trained or found on disk.")
    yield
    logger.info("Shutting down microservice.")


app = FastAPI(
    title=config.api.title,
    version=config.api.version,
    description="High-throughput NLP microservice for sentiment classification and YouTube audience analytics.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Returns microservice health, version, and model status."""
    is_ready = predictor is not None and predictor.is_ready
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        service=config.api.title,
        version=config.api.version,
        model_loaded=is_ready,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_single(request: CommentRequest):
    """
    Predicts sentiment for a single comment.
    Returns predicted class (-1, 0, 1), label, confidence score, and probability distribution.
    """
    if not predictor or not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please ensure the model is trained.",
        )
    try:
        result = predictor.predict(request.text)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(request: BatchCommentRequest):
    """
    High-throughput batch inference over multiple comments.
    Computes overall distribution and sentiment ratios.
    """
    if not predictor or not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )
    try:
        raw_results = predictor.predict_batch(request.comments)
        
        # Aggregate statistics
        distribution = {"Positive": 0, "Neutral": 0, "Negative": 0}
        parsed_results = []

        for res in raw_results:
            distribution[res["label"]] = distribution.get(res["label"], 0) + 1
            parsed_results.append(
                PredictionResponse(
                    raw_text=res["raw_text"],
                    cleaned_text=res["cleaned_text"],
                    prediction=res["prediction"],
                    label=res["label"],
                    confidence=res["confidence"],
                    probabilities=res["probabilities"],
                    latency_ms=res.get("avg_item_latency_ms", 0.0),
                )
            )

        total = len(request.comments)
        pos_ratio = round((distribution.get("Positive", 0) / max(1, total)) * 100, 2)
        batch_latency = raw_results[0].get("batch_latency_ms", 0.0) if raw_results else 0.0
        avg_latency = raw_results[0].get("avg_item_latency_ms", 0.0) if raw_results else 0.0

        return BatchPredictionResponse(
            total_comments=total,
            sentiment_distribution=distribution,
            positive_ratio=pos_ratio,
            batch_latency_ms=batch_latency,
            avg_item_latency_ms=avg_latency,
            results=parsed_results,
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post("/analyze/youtube", response_model=YouTubeAnalysisResponse, tags=["YouTube Analytics"])
async def analyze_youtube_video(request: YouTubeAnalysisRequest):
    """
    Extracts comments from a YouTube video URL and performs sentiment intelligence analysis.
    """
    if not predictor or not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    yt = YouTubeService(api_key=request.api_key) if request.api_key else youtube_service

    try:
        video_id = yt.extract_video_id(request.video_url)
        if not video_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid YouTube URL or Video ID provided.",
            )

        raw_comments = yt.get_comments(request.video_url, max_comments=request.max_comments)
        if not raw_comments:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No comments found for this video or comments are disabled.",
            )

        texts = [c["text"] for c in raw_comments]
        predictions = predictor.predict_batch(texts)

        breakdown = {"Positive": 0, "Neutral": 0, "Negative": 0}
        comment_items: List[YouTubeCommentResult] = []

        for orig_c, pred in zip(raw_comments, predictions):
            lbl = pred["label"]
            breakdown[lbl] = breakdown.get(lbl, 0) + 1
            comment_items.append(
                YouTubeCommentResult(
                    comment_id=orig_c.get("comment_id"),
                    author=orig_c.get("author", "Anonymous"),
                    text=orig_c["text"],
                    likes=orig_c.get("likes", 0),
                    published_at=orig_c.get("published_at", ""),
                    prediction=pred["prediction"],
                    label=lbl,
                    confidence=pred["confidence"],
                )
            )

        total = len(comment_items)
        percentages = {
            k: round((v / max(1, total)) * 100, 2) for k, v in breakdown.items()
        }

        # Polarity index (-100 to +100)
        pos_count = breakdown.get("Positive", 0)
        neg_count = breakdown.get("Negative", 0)
        polarity_index = round(((pos_count - neg_count) / max(1, total)) * 100, 2)

        if polarity_index > 20:
            overall = "Overwhelmingly Positive"
        elif polarity_index > 5:
            overall = "Moderately Positive"
        elif polarity_index >= -5:
            overall = "Neutral / Mixed"
        elif polarity_index >= -20:
            overall = "Moderately Negative"
        else:
            overall = "Overwhelmingly Negative"

        # Top liked positive & negative comments
        sorted_pos = sorted(
            [c for c in comment_items if c.label == "Positive"],
            key=lambda x: x.likes,
            reverse=True,
        )
        sorted_neg = sorted(
            [c for c in comment_items if c.label == "Negative"],
            key=lambda x: x.likes,
            reverse=True,
        )

        return YouTubeAnalysisResponse(
            video_id=video_id,
            total_analyzed=total,
            sentiment_breakdown=breakdown,
            sentiment_percentages=percentages,
            overall_sentiment=overall,
            positivity_index=polarity_index,
            top_positive_comment=sorted_pos[0] if sorted_pos else None,
            top_negative_comment=sorted_neg[0] if sorted_neg else None,
            comments=comment_items,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YouTube analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
