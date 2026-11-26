"""Bikolé FastAPI application. Run from the bikole/ folder:

    uvicorn backend.main:app --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from database import FeedbackRepository, MySQLConfig

from .api import router
from .config import Settings
from .nmt import NMTEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _validation_message(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "Invalid request."
    err = errors[0]
    msg = str(err.get("msg", "Invalid value.")).removeprefix("Value error, ")
    loc = [str(p) for p in err.get("loc", []) if p not in ("body",)]
    if err.get("type") == "literal_error" and loc:
        return f"{loc[-1]} must be one of: eng, bcl_South, bcl_North."
    if err.get("type") == "missing" and loc:
        return f"{loc[-1]} is required."
    if err.get("type") == "string_too_long" and loc:
        return f"{loc[-1]} is too long ({err.get('ctx', {}).get('max_length')} characters max)."
    return msg


def mysql_config(settings: Settings) -> MySQLConfig:
    return MySQLConfig(host=settings.mysql_host, port=settings.mysql_port, user=settings.mysql_user,
                       password=settings.mysql_password, database=settings.mysql_database)


def create_app(settings: Settings | None = None, engine: NMTEngine | None = None,
               repo: FeedbackRepository | None = None) -> FastAPI:
    settings = settings or Settings()
    engine = engine or NMTEngine(settings.model_dir, settings.device, settings.num_beams, settings.quantize_cpu,
                                 settings.cpu_threads)
    repo = repo or FeedbackRepository(mysql_config(settings))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not repo.try_init():   # the API still starts; feedback reports the problem until MySQL is reachable
            logging.getLogger("bikole").warning("Feedback database unavailable: %s", repo.error)
        if settings.load_model and engine.state.value == "not_loaded":
            engine.load_in_background()
        yield

    app = FastAPI(title="Bikolé NMT API", version="1.0.0", lifespan=lifespan,
                  description="English ↔ Virac (Southern Bikol) ↔ Pandan (Northern Bikol) translation.")
    app.state.engine = engine
    app.state.feedback_repo = repo

    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.exception_handler(RequestValidationError)
    async def _on_validation(_: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"status": "failed", "error": _validation_message(exc)})

    @app.exception_handler(HTTPException)
    async def _on_http(_: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"status": "failed", "error": str(exc.detail)})

    app.include_router(router)
    if settings.frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    return app


app = create_app()
