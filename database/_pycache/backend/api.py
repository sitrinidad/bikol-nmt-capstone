"""REST endpoints: /api/v1/translate, /api/v1/feedback, /api/v1/languages, /api/v1/health."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from database import DatabaseUnavailable, FeedbackRepository

from .languages import LANGUAGES
from .nmt import ModelNotReady, NMTEngine, UnsupportedDirection
from .schemas import ErrorResponse, FeedbackRequest, FeedbackResponse, TranslateRequest, TranslateResponse

log = logging.getLogger("bikole.api")
router = APIRouter(prefix="/api/v1")

ERRORS = {400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}


def _engine(request: Request) -> NMTEngine:
    return request.app.state.engine


def _repo(request: Request) -> FeedbackRepository:
    return request.app.state.feedback_repo


@router.get("/health")
def health(request: Request):
    return {"status": "ok", "model": _engine(request).info(), "database": _repo(request).info()}


@router.get("/languages")
def languages(request: Request):
    engine = _engine(request)
    return {
        "languages": [{"code": code, **meta} for code, meta in LANGUAGES.items()],
        "directions": [list(d) for d in engine.directions],
    }


@router.post("/translate", response_model=TranslateResponse, responses=ERRORS)
def translate(payload: TranslateRequest, request: Request):
    # plain `def`: FastAPI runs it in a worker thread, so model inference never blocks the server
    engine = _engine(request)
    if not engine.supports(payload.source_lang, payload.target_lang):
        raise HTTPException(400, f"Translation from {payload.source_lang} to {payload.target_lang} is not supported by the model.")
    try:
        result = engine.translate(payload.source_text, payload.source_lang, payload.target_lang)
    except ModelNotReady as exc:
        raise HTTPException(503, str(exc)) from exc
    except UnsupportedDirection as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        log.exception("Translation failed")
        raise HTTPException(500, "The model could not translate this text. Please try again.") from exc
    return TranslateResponse(
        translated_text=result.translated_text,
        confidence_score=result.confidence_score,
        inference_latency_ms=result.inference_latency_ms,
    )


@router.post("/feedback", response_model=FeedbackResponse, status_code=201, responses=ERRORS)
def feedback(payload: FeedbackRequest, request: Request):
    try:
        feedback_id = _repo(request).add(
            source_text=payload.source_text,
            source_lang=payload.source_lang,
            target_lang=payload.target_lang,
            generated_translation=payload.generated_translation,
            corrected_translation=payload.corrected_translation,
            explanation=payload.explanation,
            model_version=_engine(request).model_version,
        )
    except DatabaseUnavailable as exc:
        raise HTTPException(503, f"Your feedback could not be saved: {exc}") from exc
    except Exception as exc:
        log.exception("Saving feedback failed")
        raise HTTPException(500, "Your feedback could not be saved. Please try again.") from exc
    return FeedbackResponse(feedback_id=feedback_id, message="Thank you! Your correction was saved for future model training.")
