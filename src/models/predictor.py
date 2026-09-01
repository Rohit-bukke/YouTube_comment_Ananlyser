"""
Production Inference Engine with Confidence Scoring, Latency Tracking, and Explainability.
"""

import os
import time
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np

from src.data.preprocessor import TextPreprocessor
from src.utils.config_manager import AppConfig, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentPredictor:
    """
    Thread-safe inference engine for single and batch sentiment classification.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[AppConfig] = None,
    ):
        self.config = config or load_config()
        self.model_path = model_path or self.config.model.model_path
        self.class_names = self.config.model.class_names
        self.preprocessor = TextPreprocessor(self.config)
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Loads the serialized scikit-learn Pipeline."""
        if not os.path.exists(self.model_path):
            logger.warning(
                f"Model file {self.model_path} not found. Predictor initialized without trained model."
            )
            return

        try:
            self.pipeline = joblib.load(self.model_path)
            logger.info(f"Loaded production model pipeline from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model pipeline: {e}")
            raise e

    @property
    def is_ready(self) -> bool:
        """Returns True if the pipeline is loaded and ready for inference."""
        return self.pipeline is not None

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Executes real-time inference for a single comment.
        """
        if not self.is_ready:
            raise RuntimeError("Model pipeline is not loaded. Train the model first.")

        start_time = time.perf_counter()

        cleaned_text = self.preprocessor.clean_text(text)
        if not cleaned_text.strip():
            # Fallback for empty or purely symbol text
            return {
                "raw_text": text,
                "cleaned_text": "",
                "prediction": 0,
                "label": self.class_names.get(0, "Neutral"),
                "confidence": 0.50,
                "probabilities": {
                    self.class_names.get(k, str(k)): 1.0 / len(self.class_names)
                    for k in self.class_names
                },
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 3),
            }

        classifier = self.pipeline.named_steps["classifier"]
        vectorizer = self.pipeline.named_steps["vectorizer"]

        X_vec = vectorizer.transform([cleaned_text])
        pred_class = int(classifier.predict(X_vec)[0])

        probabilities: Dict[str, float] = {}
        confidence = 1.0

        if hasattr(classifier, "predict_proba"):
            probs = classifier.predict_proba(X_vec)[0]
            classes = classifier.classes_
            for cls_id, prob in zip(classes, probs):
                label_name = self.class_names.get(int(cls_id), str(cls_id))
                probabilities[label_name] = round(float(prob), 4)
            confidence = round(float(np.max(probs)), 4)
        else:
            # For models without predict_proba
            label_name = self.class_names.get(pred_class, str(pred_class))
            probabilities[label_name] = 1.0

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 3)

        return {
            "raw_text": text,
            "cleaned_text": cleaned_text,
            "prediction": pred_class,
            "label": self.class_names.get(pred_class, str(pred_class)),
            "confidence": confidence,
            "probabilities": probabilities,
            "latency_ms": elapsed_ms,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Executes high-throughput batch prediction over a list of texts.
        """
        if not self.is_ready:
            raise RuntimeError("Model pipeline is not loaded. Train the model first.")

        if not texts:
            return []

        start_time = time.perf_counter()

        cleaned_texts = [self.preprocessor.clean_text(t) for t in texts]
        non_empty_indices = [i for i, t in enumerate(cleaned_texts) if t.strip()]

        vectorizer = self.pipeline.named_steps["vectorizer"]
        classifier = self.pipeline.named_steps["classifier"]

        results: List[Dict[str, Any]] = [None] * len(texts)

        # Handle non-empty texts in batch
        if non_empty_indices:
            valid_texts = [cleaned_texts[i] for i in non_empty_indices]
            X_vec = vectorizer.transform(valid_texts)
            preds = classifier.predict(X_vec)

            has_proba = hasattr(classifier, "predict_proba")
            probs_matrix = classifier.predict_proba(X_vec) if has_proba else None
            classes = classifier.classes_ if has_proba else []

            for idx_pos, orig_idx in enumerate(non_empty_indices):
                p_cls = int(preds[idx_pos])
                label = self.class_names.get(p_cls, str(p_cls))
                probs_dict = {}
                conf = 1.0

                if has_proba:
                    sample_probs = probs_matrix[idx_pos]
                    for c_id, prob in zip(classes, sample_probs):
                        lbl = self.class_names.get(int(c_id), str(c_id))
                        probs_dict[lbl] = round(float(prob), 4)
                    conf = round(float(np.max(sample_probs)), 4)
                else:
                    probs_dict[label] = 1.0

                results[orig_idx] = {
                    "raw_text": texts[orig_idx],
                    "cleaned_text": cleaned_texts[orig_idx],
                    "prediction": p_cls,
                    "label": label,
                    "confidence": conf,
                    "probabilities": probs_dict,
                }

        # Fill empty items
        for i in range(len(texts)):
            if results[i] is None:
                results[i] = {
                    "raw_text": texts[i],
                    "cleaned_text": "",
                    "prediction": 0,
                    "label": self.class_names.get(0, "Neutral"),
                    "confidence": 0.50,
                    "probabilities": {
                        self.class_names.get(k, str(k)): 1.0 / len(self.class_names)
                        for k in self.class_names
                    },
                }

        batch_latency = round((time.perf_counter() - start_time) * 1000, 3)
        avg_per_item = round(batch_latency / max(1, len(texts)), 3)

        for res in results:
            res["batch_latency_ms"] = batch_latency
            res["avg_item_latency_ms"] = avg_per_item

        return results
