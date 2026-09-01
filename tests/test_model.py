"""Unit tests for Model Evaluator and SentimentPredictor."""

import os
import pytest
import numpy as np
from src.models.evaluate import ModelEvaluator
from src.models.predictor import SentimentPredictor


class TestModelEvaluator:
    """Tests for evaluation metrics computation."""

    def test_evaluate_perfect_predictions(self):
        evaluator = ModelEvaluator()
        y_true = [-1, 0, 1, -1, 0, 1]
        y_pred = [-1, 0, 1, -1, 0, 1]
        metrics = evaluator.evaluate(y_true, y_pred, model_name="PerfectModel")
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_evaluate_returns_required_keys(self):
        evaluator = ModelEvaluator()
        y_true = [1, 0, -1, 1]
        y_pred = [1, 0, 0, -1]
        metrics = evaluator.evaluate(y_true, y_pred)
        required_keys = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro", "confusion_matrix"]
        for key in required_keys:
            assert key in metrics

    def test_evaluate_metrics_range(self):
        evaluator = ModelEvaluator()
        y_true = [1, 0, -1, 1, 0]
        y_pred = [1, -1, -1, 0, 0]
        metrics = evaluator.evaluate(y_true, y_pred)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_macro"] <= 1.0

    def test_confusion_matrix_shape(self):
        evaluator = ModelEvaluator()
        y_true = [1, 0, -1]
        y_pred = [1, 0, -1]
        metrics = evaluator.evaluate(y_true, y_pred)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)


class TestSentimentPredictor:
    """Tests for SentimentPredictor inference engine."""

    @pytest.fixture
    def predictor(self):
        model_path = "models/sentiment_pipeline.joblib"
        if not os.path.exists(model_path):
            pytest.skip("Trained model not found. Run training pipeline first.")
        return SentimentPredictor(model_path=model_path)

    def test_is_ready(self, predictor):
        assert predictor.is_ready is True

    def test_predict_returns_required_keys(self, predictor):
        result = predictor.predict("This is an amazing video!")
        required_keys = ["raw_text", "cleaned_text", "prediction", "label", "confidence", "probabilities", "latency_ms"]
        for key in required_keys:
            assert key in result

    def test_predict_label_values(self, predictor):
        result = predictor.predict("Absolutely terrible content")
        assert result["label"] in ["Positive", "Neutral", "Negative"]
        assert result["prediction"] in [-1, 0, 1]

    def test_predict_confidence_range(self, predictor):
        result = predictor.predict("Great explanation of machine learning")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_empty_text(self, predictor):
        result = predictor.predict("")
        assert result["label"] == "Neutral"
        assert result["confidence"] == 0.50

    def test_predict_batch(self, predictor):
        comments = [
            "Loved this tutorial!",
            "What library did you use?",
            "Horrible quality and wrong info.",
        ]
        results = predictor.predict_batch(comments)
        assert len(results) == 3
        for r in results:
            assert "label" in r
            assert "confidence" in r

    def test_predict_batch_empty_list(self, predictor):
        results = predictor.predict_batch([])
        assert results == []

    def test_predictor_without_model(self):
        pred = SentimentPredictor(model_path="nonexistent_model.joblib")
        assert pred.is_ready is False
        with pytest.raises(RuntimeError, match="not loaded"):
            pred.predict("test")
