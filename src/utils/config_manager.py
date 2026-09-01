"""
Configuration management utilities using Pydantic dataclasses and YAML loader.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataConfig:
    dataset_url: str = "https://raw.githubusercontent.com/entbappy/End-to-end-Youtube-Sentiment/refs/heads/main/notebooks/reddit_preprocessing.csv"
    raw_data_path: str = "data/raw/raw_comments.csv"
    processed_data_path: str = "data/processed/clean_comments.parquet"
    text_column: str = "clean_comment"
    target_column: str = "category"
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class PreprocessingConfig:
    remove_urls: bool = True
    remove_mentions: bool = True
    remove_special_chars: bool = True
    lowercase: bool = True
    expand_contractions: bool = True
    lemmatize: bool = True
    remove_stopwords: bool = True


@dataclass
class FeaturesConfig:
    ngram_range: List[int] = field(default_factory=lambda: [1, 2])
    max_features: int = 25000
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True


@dataclass
class ModelConfig:
    target_metric: str = "f1_macro"
    model_dir: str = "models"
    model_path: str = "models/sentiment_pipeline.joblib"
    metrics_path: str = "models/metrics.json"
    class_names: Dict[int, str] = field(
        default_factory=lambda: {-1: "Negative", 0: "Neutral", 1: "Positive"}
    )


@dataclass
class APIConfig:
    title: str = "YouTube Comment Sentiment Analyzer API"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class StreamlitConfig:
    title: str = "YouTube Sentiment & Audience Intelligence"
    page_icon: str = "📊"


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    app: StreamlitConfig = field(default_factory=StreamlitConfig)


def load_config(config_path: str = "configs/config.yaml") -> AppConfig:
    """
    Loads YAML configuration and initializes AppConfig.
    Falls back to default config if file is missing or invalid.
    """
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found. Using default configuration.")
        return AppConfig()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}

        data_cfg = DataConfig(**raw_dict.get("data", {}))
        prep_cfg = PreprocessingConfig(**raw_dict.get("preprocessing", {}))
        feat_cfg = FeaturesConfig(**raw_dict.get("features", {}))
        
        # Ensure class_names have integer keys
        model_dict = raw_dict.get("model", {})
        if "class_names" in model_dict and isinstance(model_dict["class_names"], dict):
            model_dict["class_names"] = {int(k): v for k, v in model_dict["class_names"].items()}
        model_cfg = ModelConfig(**model_dict)
        
        api_cfg = APIConfig(**raw_dict.get("api", {}))
        app_cfg = StreamlitConfig(**raw_dict.get("app", {}))

        return AppConfig(
            data=data_cfg,
            preprocessing=prep_cfg,
            features=feat_cfg,
            model=model_cfg,
            api=api_cfg,
            app=app_cfg,
        )
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}. Falling back to default.")
        return AppConfig()
