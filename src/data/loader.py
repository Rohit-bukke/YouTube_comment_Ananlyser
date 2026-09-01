"""
Data ingestion and loading module with caching and validation support.
"""

import os
from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config_manager import AppConfig, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Handles downloading, loading, splitting, and caching datasets.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()

    def fetch_raw_data(self, force_download: bool = False) -> pd.DataFrame:
        """
        Loads raw dataset from local cache or downloads from remote URL.
        """
        raw_path = self.config.data.raw_data_path
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)

        if os.path.exists(raw_path) and not force_download:
            logger.info(f"Loading raw dataset from local cache: {raw_path}")
            df = pd.read_csv(raw_path)
        else:
            url = self.config.data.dataset_url
            logger.info(f"Downloading raw dataset from: {url}")
            df = pd.read_csv(url)
            df.to_csv(raw_path, index=False)
            logger.info(f"Saved raw dataset to: {raw_path}")

        logger.info(f"Raw dataset shape: {df.shape}")
        return df

    def validate_and_clean_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates schema, drops missing/corrupt values, and removes exact duplicates.
        """
        text_col = self.config.data.text_column
        target_col = self.config.data.target_column

        if text_col not in df.columns or target_col not in df.columns:
            raise ValueError(f"Dataset must contain '{text_col}' and '{target_col}' columns.")

        initial_len = len(df)
        df = df[[text_col, target_col]].dropna()
        df[text_col] = df[text_col].astype(str)
        df[target_col] = df[target_col].astype(int)

        # Filter empty strings or whitespaces
        df = df[df[text_col].str.strip() != ""]
        df = df.drop_duplicates(subset=[text_col])

        logger.info(f"Data validation complete: {initial_len} -> {len(df)} records ({initial_len - len(df)} removed).")
        return df

    def load_and_split(
        self, force_download: bool = False
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Loads validated data and splits into train and test sets.
        """
        df = self.fetch_raw_data(force_download=force_download)
        df = self.validate_and_clean_schema(df)

        text_col = self.config.data.text_column
        target_col = self.config.data.target_column

        X = df[text_col]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_state,
            stratify=y,
        )

        logger.info(f"Dataset split: Train={len(X_train)} samples, Test={len(X_test)} samples.")
        return X_train, X_test, y_train, y_test
