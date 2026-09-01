"""Setup configuration for YouTube Comment Sentiment Analyzer."""

from setuptools import setup, find_packages

setup(
    name="youtube-comment-sentiment-analyzer",
    version="1.0.0",
    description="Production-grade YouTube Comment Sentiment Analysis & NLP Intelligence System",
    author="Rohit Bukke",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.4",
        "scikit-learn>=1.2",
        "nltk>=3.8",
        "matplotlib>=3.5",
        "seaborn>=0.12",
        "fastapi>=0.100",
        "uvicorn>=0.20",
        "pydantic>=2.0",
        "streamlit>=1.28",
        "PyYAML>=6.0",
        "requests>=2.28",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "httpx>=0.24"],
    },
)
