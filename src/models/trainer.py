"""
Multi-Model Training & Benchmarking Pipeline with Automated Best-Model Selection.
"""

import os
from typing import Any, Dict, Optional, Tuple
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data.loader import DataLoader
from src.data.preprocessor import TextPreprocessor
from src.features.vectorizer import FeatureVectorizer
from src.models.evaluate import ModelEvaluator
from src.utils.config_manager import AppConfig, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Trains, benchmarks, and serializes machine learning models for sentiment analysis.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.evaluator = ModelEvaluator(class_names=self.config.model.class_names)
        self.preprocessor = TextPreprocessor(self.config)
        self.vectorizer = FeatureVectorizer(self.config)

    def get_candidate_models(self) -> Dict[str, Any]:
        """
        Returns a dictionary of candidate classification models.
        """
        return {
            "Multinomial Naive Bayes": MultinomialNB(alpha=0.2),
            "Logistic Regression": LogisticRegression(
                C=1.5,
                max_iter=1000,
                class_weight="balanced",
                random_state=self.config.data.random_state,
            ),
            "Linear SVC (Calibrated)": CalibratedClassifierCV(
                estimator=LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    random_state=self.config.data.random_state,
                    max_iter=2000,
                ),
                cv=3,
            ),
            "SGD Classifier": SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                penalty="l2",
                class_weight="balanced",
                random_state=self.config.data.random_state,
            ),
        }

    def train_and_benchmark(
        self, force_download: bool = False
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Loads data, preprocesses, trains multiple candidate models, evaluates them,
        and saves the top-performing pipeline.
        """
        logger.info("=== Starting Model Training & Benchmarking Pipeline ===")
        loader = DataLoader(self.config)
        X_train_raw, X_test_raw, y_train, y_test = loader.load_and_split(
            force_download=force_download
        )

        logger.info("Preprocessing training and testing text...")
        X_train_clean = self.preprocessor.transform(X_train_raw)
        X_test_clean = self.preprocessor.transform(X_test_raw)

        logger.info("Extracting TF-IDF features...")
        X_train_vec = self.vectorizer.fit_transform(X_train_clean)
        X_test_vec = self.vectorizer.transform(X_test_clean)

        candidate_models = self.get_candidate_models()

        # Add Voting Ensemble
        voting_ensemble = VotingClassifier(
            estimators=[
                ("nb", candidate_models["Multinomial Naive Bayes"]),
                ("lr", candidate_models["Logistic Regression"]),
                ("svc", candidate_models["Linear SVC (Calibrated)"]),
            ],
            voting="soft",
        )
        candidate_models["Soft Voting Ensemble"] = voting_ensemble

        benchmark_results = {}
        best_model_name = None
        best_model = None
        best_score = -1.0
        target_metric = self.config.model.target_metric

        for name, model in candidate_models.items():
            logger.info(f"--- Training candidate model: {name} ---")
            model.fit(X_train_vec, y_train)

            y_pred = model.predict(X_test_vec)
            metrics = self.evaluator.evaluate(y_test, y_pred, model_name=name)

            # Benchmark latency
            sample_size = min(200, X_test_vec.shape[0])
            latency_stats = self.evaluator.benchmark_latency(model, X_test_vec[:sample_size])
            metrics["latency"] = latency_stats

            benchmark_results[name] = metrics

            score = metrics.get(target_metric, metrics["accuracy"])
            if score > best_score:
                best_score = score
                best_model_name = name
                best_model = model

        logger.info(f"[BEST MODEL] {best_model_name} with {target_metric}={best_score:.4f}")

        # Assemble end-to-end production scikit-learn Pipeline
        best_pipeline = Pipeline([
            ("vectorizer", self.vectorizer.vectorizer),
            ("classifier", best_model),
        ])

        # Serialize artifacts
        os.makedirs(self.config.model.model_dir, exist_ok=True)
        pipeline_path = self.config.model.model_path
        joblib.dump(best_pipeline, pipeline_path)
        logger.info(f"Serialized production pipeline to: {pipeline_path}")

        # Compile final benchmark metadata
        final_metadata = {
            "best_model": best_model_name,
            "target_metric": target_metric,
            "best_score": round(float(best_score), 4),
            "train_samples": len(X_train_raw),
            "test_samples": len(X_test_raw),
            "vocabulary_size": len(self.vectorizer.vectorizer.vocabulary_),
            "models": benchmark_results,
        }

        self.evaluator.save_metrics(final_metadata, self.config.model.metrics_path)
        logger.info("=== Training and Benchmarking Complete ===")
        return best_pipeline, final_metadata


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_and_benchmark()
