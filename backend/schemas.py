"""Request / response models for the REST API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .config import MAX_EXPLANATION_CHARS, MAX_SOURCE_CHARS, MAX_TRANSLATION_CHARS

LangCode = Literal["eng", "bcl_South", "bcl_North"]


def _required_text(value: str, field_name: str) -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty.")
    return value


class _LangPair(BaseModel):
    source_lang: LangCode
    target_lang: LangCode

    @model_validator(mode="after")
    def _different_languages(self):
        if self.source_lang == self.target_lang:
            raise ValueError("Source and target language must be different.")
        return self


class TranslateRequest(_LangPair):
    source_text: str = Field(..., max_length=MAX_SOURCE_CHARS)

    @field_validator("source_text")
    @classmethod
    def _text(cls, v: str) -> str:
        return _required_text(v, "Source text")


class TranslateResponse(BaseModel):
    status: Literal["success"] = "success"
    translated_text: str
    confidence_score: Optional[float]
    inference_latency_ms: int


class FeedbackRequest(_LangPair):
    source_text: str = Field(..., max_length=MAX_SOURCE_CHARS)
    generated_translation: str = Field(..., max_length=MAX_TRANSLATION_CHARS)
    corrected_translation: str = Field(..., max_length=MAX_TRANSLATION_CHARS)
    explanation: Optional[str] = Field(default=None, max_length=MAX_EXPLANATION_CHARS)

    @field_validator("source_text")
    @classmethod
    def _src(cls, v: str) -> str:
        return _required_text(v, "Source text")

    @field_validator("generated_translation")
    @classmethod
    def _gen(cls, v: str) -> str:
        return _required_text(v, "Generated translation")

    @field_validator("corrected_translation")
    @classmethod
    def _cor(cls, v: str) -> str:
        return _required_text(v, "Corrected translation")

    @field_validator("explanation")
    @classmethod
    def _exp(cls, v: Optional[str]) -> Optional[str]:
        v = (v or "").strip()
        return v or None


class FeedbackResponse(BaseModel):
    status: Literal["success"] = "success"
    feedback_id: int
    message: str


class ErrorResponse(BaseModel):
    status: Literal["failed"] = "failed"
    error: str
