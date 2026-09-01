"""
Model Evaluation and Benchmarking Engine with detailed statistical metrics and latency profiling.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates ML models for multi-class classification and benchmarks performance.
    """

    def __init__(self, class_names: Optional[Dict[int, str]] = None):
        self.class_names = class_names or {-1: "Negative", 0: "Neutral", 1: "Positive"}
        self.labels = sorted(list(self.class_names.keys()))
        self.target_names = [self.class_names[k] for k in self.labels]

    def evaluate(self, y_true, y_pred, model_name: str = "Model") -> Dict[str, Any]:
        """
        Computes precision, recall, f1-score, accuracy, and confusion matrix.
        """
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=self.labels).tolist()
        report = classification_report(
            y_true,
            y_pred,
            labels=self.labels,
            target_names=self.target_names,
            output_dict=True,
            zero_division=0,
        )

        metrics = {
            "model_name": model_name,
            "accuracy": round(float(acc), 4),
            "f1_macro": round(float(f1_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "precision_macro": round(float(precision_macro), 4),
            "recall_macro": round(float(recall_macro), 4),
            "confusion_matrix": cm,
            "labels": self.labels,
            "target_names": self.target_names,
            "classification_report": report,
        }

        logger.info(
            f"[{model_name}] Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f} | Weighted F1: {f1_weighted:.4f}"
        )
        return metrics

    def benchmark_latency(
        self, model, X_sample, n_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Measures inference latency in milliseconds per sample.
        """
        n_samples = X_sample.shape[0] if hasattr(X_sample, "shape") else len(X_sample)
        latencies: List[float] = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = model.predict(X_sample)
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0 / max(1, n_samples))

        latencies_arr = np.array(latencies)
        return {
            "avg_latency_ms": round(float(np.mean(latencies_arr)), 3),
            "p50_latency_ms": round(float(np.median(latencies_arr)), 3),
            "p95_latency_ms": round(float(np.percentile(latencies_arr, 95)), 3),
        }

    def save_metrics(self, metrics: Dict[str, Any], filepath: str) -> None:
        """Saves evaluation results to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Evaluation metrics saved to {filepath}")
