"""Unit tests for the FastAPI application endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data


class TestPredictEndpoint:
    def test_predict_single_valid(self, client):
        response = client.post("/predict", json={"text": "This is a wonderful tutorial!"})
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        assert response.status_code == 200
        data = response.json()
        assert data["label"] in ["Positive", "Neutral", "Negative"]
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_single_empty_text(self, client):
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422  # Pydantic validation error

    def test_predict_missing_field(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    def test_batch_predict_valid(self, client):
        payload = {
            "comments": [
                "Amazing content!",
                "Terrible video.",
                "What is the dataset source?",
            ]
        }
        response = client.post("/predict/batch", json=payload)
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        assert response.status_code == 200
        data = response.json()
        assert data["total_comments"] == 3
        assert len(data["results"]) == 3
        assert "sentiment_distribution" in data


class TestYouTubeEndpoint:
    def test_youtube_analysis_valid(self, client):
        payload = {
            "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "max_comments": 10,
        }
        response = client.post("/analyze/youtube", json=payload)
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        assert response.status_code == 200
        data = response.json()
        assert "video_id" in data
        assert "sentiment_breakdown" in data
        assert "positivity_index" in data

    def test_youtube_invalid_url(self, client):
        payload = {"video_url": "not_a_url", "max_comments": 10}
        response = client.post("/analyze/youtube", json=payload)
        # Should return 400 or 500 depending on extraction
        assert response.status_code in [400, 500, 503]
