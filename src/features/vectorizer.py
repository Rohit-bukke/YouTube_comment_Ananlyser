"""
Feature Extraction and Vectorization Engine with Sublinear TF-IDF scaling.
"""

import os
from typing import Dict, Iterable, List, Optional, Tuple
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.config_manager import AppConfig, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureVectorizer:
    """
    Sublinear TF-IDF feature extractor with n-gram support and model serialization.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        feat_cfg = self.config.features

        ngram_range = tuple(feat_cfg.ngram_range) if isinstance(feat_cfg.ngram_range, list) else (1, 2)

        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=feat_cfg.max_features,
            min_df=feat_cfg.min_df,
            max_df=feat_cfg.max_df,
            sublinear_tf=feat_cfg.sublinear_tf,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.is_fitted = False

    def fit(self, texts: Iterable[str]) -> "FeatureVectorizer":
        """Fits the TF-IDF vectorizer on training text corpus."""
        logger.info("Fitting TF-IDF Vectorizer...")
        self.vectorizer.fit(texts)
        self.is_fitted = True
        logger.info(f"Vectorizer fitted successfully. Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        return self

    def transform(self, texts: Iterable[str]):
        """Transforms text corpus into TF-IDF sparse matrix."""
        if not self.is_fitted:
            raise ValueError("FeatureVectorizer must be fitted before transforming data.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: Iterable[str]):
        """Fits vectorizer and transforms text corpus."""
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self) -> List[str]:
        """Returns array of feature names / terms."""
        return self.vectorizer.get_feature_names_out().tolist()

    def get_top_features(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Returns the top N features with highest average IDF weights."""
        if not self.is_fitted:
            return []
        feature_names = self.get_feature_names()
        idf_scores = self.vectorizer.idf_
        sorted_indices = idf_scores.argsort()[::-1][:top_n]
        return [(feature_names[i], float(idf_scores[i])) for i in sorted_indices]

    def save(self, filepath: str) -> None:
        """Saves the vectorizer artifact to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)
        logger.info(f"Vectorizer saved to {filepath}")

    def load(self, filepath: str) -> "FeatureVectorizer":
        """Loads a saved vectorizer artifact from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vectorizer file {filepath} not found.")
        self.vectorizer = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Vectorizer loaded from {filepath}")
        return self
