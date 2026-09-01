"""
Text Preprocessing Engine for Social Media & YouTube Comments.
Features URL/mention cleaning, contraction expansion, sentiment-aware stopword filtering, and lemmatization.
"""

import html
import re
from typing import Iterable, List, Optional, Union
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.utils.config_manager import AppConfig, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Common English Contractions Mapping
CONTRACTIONS = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

# Negation words to retain (crucial for sentiment polarity preservation)
NEGATION_WORDS = {
    "not", "no", "nor", "neither", "never", "none", "hardly", "barely",
    "scarcely", "without", "against", "cannot", "won't", "don't", "didn't",
    "isn't", "wasn't", "shouldn't", "couldn't", "wouldn't", "hasn't", "haven't", "hadn't"
}


class TextPreprocessor:
    """
    Production-ready text preprocessing pipeline tailored for YouTube comment analytics.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()

        # Regex patterns
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.html_tag_pattern = re.compile(r"<.*?>")
        self.mention_pattern = re.compile(r"@\w+")
        self.hashtag_pattern = re.compile(r"#(\w+)")
        self.alpha_pattern = re.compile(r"[^a-zA-Z\s]")
        self.whitespace_pattern = re.compile(r"\s+")
        self.contraction_pattern = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS.keys()) + r")\b",
            re.IGNORECASE,
        )

        # Stopwords initialization (preserving negations)
        try:
            raw_stops = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            raw_stops = set(stopwords.words("english"))
        
        self.stop_words = raw_stops - NEGATION_WORDS

        # Lemmatizer initialization
        try:
            self.lemmatizer = WordNetLemmatizer()
            # Warm up
            _ = self.lemmatizer.lemmatize("testing")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            self.lemmatizer = WordNetLemmatizer()

    def expand_contractions(self, text: str) -> str:
        """Expands common English contractions."""
        def replace(match):
            word = match.group(0).lower()
            return CONTRACTIONS.get(word, word)
        return self.contraction_pattern.sub(replace, text)

    def clean_text(self, text: Union[str, float, None]) -> str:
        """
        Cleans a single input text string.
        """
        if text is None or not isinstance(text, str):
            return ""

        # 1. Unescape HTML entities (e.g., &amp; -> &)
        text = html.unescape(text)

        # 2. Lowercase
        if self.config.preprocessing.lowercase:
            text = text.lower()

        # 3. Remove URLs
        if self.config.preprocessing.remove_urls:
            text = self.url_pattern.sub(" ", text)

        # 4. Remove HTML tags
        text = self.html_tag_pattern.sub(" ", text)

        # 5. Remove @mentions
        if self.config.preprocessing.remove_mentions:
            text = self.mention_pattern.sub(" ", text)

        # 6. Normalize hashtags (#awesome -> awesome)
        text = self.hashtag_pattern.sub(r"\1", text)

        # 7. Expand contractions
        if self.config.preprocessing.expand_contractions:
            text = self.expand_contractions(text)

        # 8. Remove special characters and digits (retain alphabets and spaces)
        if self.config.preprocessing.remove_special_chars:
            text = self.alpha_pattern.sub(" ", text)

        # 9. Tokenization, stopword removal & lemmatization
        tokens = text.split()
        cleaned_tokens: List[str] = []

        for token in tokens:
            if self.config.preprocessing.remove_stopwords and token in self.stop_words:
                continue
            if len(token) <= 1:
                continue
            if self.config.preprocessing.lemmatize:
                token = self.lemmatizer.lemmatize(token)
            cleaned_tokens.append(token)

        result = " ".join(cleaned_tokens)
        return self.whitespace_pattern.sub(" ", result).strip()

    def transform(self, texts: Union[Iterable[str], str]) -> Union[List[str], str]:
        """
        Transforms a string or list/iterable of strings.
        """
        if isinstance(texts, str):
            return self.clean_text(texts)
        return [self.clean_text(t) for t in texts]

    def __call__(self, text: str) -> str:
        return self.clean_text(text)
