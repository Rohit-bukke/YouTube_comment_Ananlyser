"""Unit tests for the TextPreprocessor module."""

import pytest
from src.data.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


class TestTextPreprocessor:
    """Tests for text cleaning and preprocessing logic."""

    def test_url_removal(self, preprocessor):
        text = "Check this out https://example.com/path and http://test.org"
        result = preprocessor.clean_text(text)
        assert "https" not in result
        assert "http" not in result
        assert "example" not in result

    def test_mention_removal(self, preprocessor):
        text = "Thanks @username for the great video @another_user"
        result = preprocessor.clean_text(text)
        assert "@username" not in result
        assert "@another_user" not in result

    def test_html_entity_decoding(self, preprocessor):
        text = "This is &amp; that is &lt;good&gt;"
        result = preprocessor.clean_text(text)
        assert "&amp;" not in result
        assert "&lt;" not in result

    def test_contraction_expansion(self, preprocessor):
        text = "I can't believe it won't work"
        result = preprocessor.clean_text(text)
        assert "can't" not in result
        assert "won't" not in result

    def test_negation_preservation(self, preprocessor):
        text = "I do not like this video at all"
        result = preprocessor.clean_text(text)
        assert "not" in result

    def test_lowercase(self, preprocessor):
        text = "THIS IS UPPERCASE TEXT"
        result = preprocessor.clean_text(text)
        assert result == result.lower()

    def test_hashtag_normalization(self, preprocessor):
        text = "This is #awesome content"
        result = preprocessor.clean_text(text)
        assert "#" not in result
        assert "awesome" in result

    def test_empty_string_handling(self, preprocessor):
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text(None) == ""
        assert preprocessor.clean_text("   ") == ""

    def test_special_chars_removal(self, preprocessor):
        text = "Great video!!! 100% recommended??? $$$"
        result = preprocessor.clean_text(text)
        assert "!" not in result
        assert "%" not in result
        assert "?" not in result
        assert "$" not in result

    def test_whitespace_normalization(self, preprocessor):
        text = "too    many     spaces   here"
        result = preprocessor.clean_text(text)
        assert "  " not in result

    def test_transform_single_string(self, preprocessor):
        result = preprocessor.transform("hello world")
        assert isinstance(result, str)

    def test_transform_list(self, preprocessor):
        result = preprocessor.transform(["hello world", "test text"])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_callable(self, preprocessor):
        result = preprocessor("hello world")
        assert isinstance(result, str)

    def test_short_token_removal(self, preprocessor):
        text = "I a am good"
        result = preprocessor.clean_text(text)
        # Single-char tokens like "I" and "a" should be removed
        tokens = result.split()
        assert all(len(t) > 1 for t in tokens)

    def test_newline_handling(self, preprocessor):
        text = "first line\nsecond line\r\nthird line"
        result = preprocessor.clean_text(text)
        assert "\n" not in result
        assert "\r" not in result
