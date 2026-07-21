"""NLP metrics — Sentiment, Emotion, LanguageDetection, Readability, TokenCount.

All are dual (question/answer) metrics. Sentiment / Emotion are LLM-judge
classifiers (verbatim prompts). LanguageDetection uses lingua, Readability uses
textstat, TokenCount uses tiktoken — each lazily imported / cached.
"""

from __future__ import annotations

import gc
from functools import lru_cache
from typing import Any

from .base_metric import DualTargetMetric

SENTIMENT_PROMPT = """
        You are an expert sentiment analysis AI. Your task is to analyze the sentiment of the provided text and classify it into one of the following categories: Positive, Negative, or Neutral.
        The output should be just the class and no reasoning/explanation.
        Text: {text}
        """

EMOTION_PROMPT = """
        You are an expert emotion analysis AI. Your task is to analyze the emotion of the provided text and classify it into one of the following categories: Joy, Anger, Sadness, Fear, Surprise, Disgust, Neutral. If more than emotion is possible then choose the highly likely emotion.
        The output should be just the class and no reasoning/explanation.
        Text: {text}
        """


@lru_cache(maxsize=1)
def _get_lang_detector():
    from lingua import Language, LanguageDetectorBuilder

    print("Loading language detector...")
    return LanguageDetectorBuilder.from_languages(
        Language.ARABIC,
        Language.CHINESE,
        Language.ENGLISH,
        Language.FRENCH,
        Language.GERMAN,
        Language.ITALIAN,
        Language.JAPANESE,
        Language.KOREAN,
        Language.PORTUGUESE,
        Language.RUSSIAN,
        Language.SPANISH,
        Language.THAI,
        Language.VIETNAMESE,
        Language.TAMIL,
        Language.HINDI,
    ).build()


class SentimentMetric(DualTargetMetric):
    name_suffix = "sentiment"

    async def a_measure(self, test_case: Any) -> Any:
        try:
            result = await self._arun_prompt(
                SENTIMENT_PROMPT, ["text"], {"text": self._text(test_case)}
            )
            self.score = result
        except Exception as e:  # noqa: BLE001 - mirror legacy behavior
            print("Sentiment error occurred:", str(e))
            self.score = ""
        self.is_successful()
        return self.score


class EmotionMetric(DualTargetMetric):
    name_suffix = "emotion"

    async def a_measure(self, test_case: Any) -> Any:
        try:
            result = await self._arun_prompt(
                EMOTION_PROMPT, ["text"], {"text": self._text(test_case)}
            )
            self.score = result
        except Exception as e:  # noqa: BLE001 - mirror legacy behavior
            print("Emotion error occurred:", str(e))
            self.score = ""
        self.is_successful()
        return self.score


class LanguageDetectionMetric(DualTargetMetric):
    name_suffix = "language"

    async def a_measure(self, test_case: Any) -> Any:
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            detector = _get_lang_detector()
            text = self._text(test_case)
            detected_language = await loop.run_in_executor(
                None, lambda: detector.detect_language_of(text)
            )
            if detected_language:
                self.score = str(detected_language).split(".")[1]
            else:
                self.score = None
        except Exception as e:  # noqa: BLE001 - mirror legacy behavior
            print(f"Error detecting language: {str(e)}")
            self.score = None
        self.is_successful()
        return self.score


class ReadabilityMetric(DualTargetMetric):
    """Flesch-Kincaid grade level (legacy ``text_quality``)."""

    name_suffix = "flesch_kincaid_grade"

    async def a_measure(self, test_case: Any) -> Any:
        import asyncio

        from textstat import textstat

        loop = asyncio.get_event_loop()
        text = self._text(test_case)
        self.score = await loop.run_in_executor(
            None, lambda: textstat.flesch_kincaid_grade(text)
        )
        self.is_successful()
        return self.score


class TokenCountMetric(DualTargetMetric):
    """Token count via tiktoken (legacy ``num_tokens_from_string``)."""

    name_suffix = "tokens"

    def __init__(self, model=None, threshold=None, target="actual_output",
                 encoding_name: str = "o200k_base") -> None:
        super().__init__(model=model, threshold=threshold, target=target)
        self.encoding_name = encoding_name

    async def a_measure(self, test_case: Any) -> Any:
        import tiktoken

        encoding = tiktoken.get_encoding(self.encoding_name)
        num_tokens = len(encoding.encode(self._text(test_case)))
        del encoding
        gc.collect()
        self.score = num_tokens
        self.is_successful()
        return self.score
