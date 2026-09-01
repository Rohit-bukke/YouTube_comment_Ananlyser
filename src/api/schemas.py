"""
Pydantic Request and Response Schemas for the FastAPI service.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CommentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Raw comment text to analyze",
        example="This machine learning tutorial is incredibly clear and well explained!",
    )


class BatchCommentRequest(BaseModel):
    comments: List[str] = Field(
        ...,
        min_items=1,
        description="List of comment texts to analyze in batch",
        example=[
            "Loved the walkthrough!",
            "Completely useless and buggy code.",
            "What is the time complexity of this model?",
        ],
    )


class PredictionResponse(BaseModel):
    raw_text: str
    cleaned_text: str
    prediction: int = Field(description="-1: Negative, 0: Neutral, 1: Positive")
    label: str = Field(description="Negative, Neutral, or Positive")
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    total_comments: int
    sentiment_distribution: Dict[str, int]
    positive_ratio: float
    batch_latency_ms: float
    avg_item_latency_ms: float
    results: List[PredictionResponse]


class YouTubeAnalysisRequest(BaseModel):
    video_url: str = Field(
        ...,
        description="Full YouTube video URL or 11-character video ID",
        example="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    )
    max_comments: Optional[int] = Field(
        default=50,
        ge=5,
        le=500,
        description="Maximum number of comments to fetch and analyze",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Optional YouTube Data API v3 key",
    )


class YouTubeCommentResult(BaseModel):
    comment_id: Optional[str]
    author: str
    text: str
    likes: int
    published_at: str
    prediction: int
    label: str
    confidence: float


class YouTubeAnalysisResponse(BaseModel):
    video_id: str
    total_analyzed: int
    sentiment_breakdown: Dict[str, int]
    sentiment_percentages: Dict[str, float]
    overall_sentiment: str
    positivity_index: float
    top_positive_comment: Optional[YouTubeCommentResult]
    top_negative_comment: Optional[YouTubeCommentResult]
    comments: List[YouTubeCommentResult]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
    target_metric: Optional[str] = None
    best_score: Optional[float] = None
