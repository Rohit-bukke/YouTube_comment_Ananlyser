"""Unit tests for the FeatureVectorizer module."""

import os
import tempfile
import pytest
from src.features.vectorizer import FeatureVectorizer


@pytest.fixture
def vectorizer():
    return FeatureVectorizer()


@pytest.fixture
def sample_corpus():
    return [
        "this is a great video about machine learning",
        "terrible explanation of natural language processing",
        "decent overview of deep learning concepts",
        "amazing tutorial on sentiment analysis models",
        "worst coding walkthrough ever seen online",
    ]


class TestFeatureVectorizer:
    """Tests for TF-IDF feature extraction."""

    def test_fit_sets_fitted_flag(self, vectorizer, sample_corpus):
        vectorizer.fit(sample_corpus)
        assert vectorizer.is_fitted is True

    def test_transform_before_fit_raises(self, vectorizer, sample_corpus):
        with pytest.raises(ValueError, match="must be fitted"):
            vectorizer.transform(sample_corpus)

    def test_fit_transform_returns_sparse(self, vectorizer, sample_corpus):
        X = vectorizer.fit_transform(sample_corpus)
        assert X.shape[0] == len(sample_corpus)
        assert X.shape[1] > 0

    def test_feature_names(self, vectorizer, sample_corpus):
        vectorizer.fit(sample_corpus)
        names = vectorizer.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_save_and_load(self, vectorizer, sample_corpus):
        vectorizer.fit(sample_corpus)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_vec.joblib")
            vectorizer.save(path)
            assert os.path.exists(path)

            new_vec = FeatureVectorizer()
            new_vec.load(path)
            assert new_vec.is_fitted is True

            X_orig = vectorizer.transform(sample_corpus)
            X_loaded = new_vec.transform(sample_corpus)
            assert X_orig.shape == X_loaded.shape

    def test_load_nonexistent_raises(self, vectorizer):
        with pytest.raises(FileNotFoundError):
            vectorizer.load("nonexistent_path.joblib")

    def test_get_top_features(self, vectorizer, sample_corpus):
        vectorizer.fit(sample_corpus)
        top = vectorizer.get_top_features(top_n=5)
        assert isinstance(top, list)
        assert len(top) <= 5
        assert all(isinstance(t, tuple) and len(t) == 2 for t in top)

    def test_get_top_features_before_fit(self, vectorizer):
        result = vectorizer.get_top_features()
        assert result == []
