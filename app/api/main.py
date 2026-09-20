# ### FILE: app/api/main.py
"""
FastAPI Main Application Entrypoint.
Configures Lifespan, CORS, Global Exception Handlers, Static Mounts, and Routers.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
import torch
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.core.exceptions import SmartNotesBaseException
from app.core.logging import setup_logging, get_logger
from app.db.session import init_db_schema, close_db_connections
from app.ml.weights_initializer import ensure_weights_exist
from app.api.v1.router import api_v1_router

settings = get_settings()
setup_logging(log_level=settings.LOG_LEVEL, debug=settings.DEBUG)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and graceful shutdown lifecycle manager."""
    logger.info("Initializing %s (v%s)...", settings.PROJECT_NAME, settings.PROJECT_VERSION)

    # 1. Ensure storage roots exist
    settings.STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

    # 2. Ensure model weights are present
    ensure_weights_exist(settings.ML_MODEL_WEIGHTS_PATH)

    # 3. Initialize database tables
    try:
        await init_db_schema()
    except Exception as exc:
        logger.warning("Postgres schema init deferred or failed (continuing startup): %s", exc)

    # 4. Pre-warm inference engine (TrOCR/CRNN) so first user upload is instant
    try:
        from app.api.dependencies import get_inference_engine
        engine = get_inference_engine()
        logger.info("Inference engine pre-warmed: %s", type(engine).__name__)
    except Exception as eng_exc:
        logger.warning("Inference engine pre-warming deferred: %s", eng_exc)

    logger.info("Application startup sequence complete.")
    yield

    # Teardown
    logger.info("Executing application graceful shutdown...")
    await close_db_connections()
    logger.info("Application shutdown complete.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="Интеллектуальная система распознавания и структурирования рукописных конспектов.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Artifacts for UI inspection
storage_path = settings.STORAGE_ROOT.resolve()
storage_path.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(storage_path)), name="static")


# Global Exception Handlers
@app.exception_handler(SmartNotesBaseException)
async def smart_notes_exception_handler(request: Request, exc: SmartNotesBaseException) -> JSONResponse:
    """Handle all known domain exceptions with standard error JSON."""
    logger.warning("Domain exception on %s: %s", request.url.path, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Centralized safety-net handler preventing stack traces leaking to clients."""
    logger.error("Unhandled internal server error on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected internal server error occurred. Please consult server logs.",
            "status_code": 500,
            "details": {"path": str(request.url.path)},
        },
    )


# System Health and Diagnostics
@app.get("/health", tags=["System Diagnostics"])
async def health_check() -> dict:
    """Liveness and readiness health probe."""
    return {
        "status": "healthy",
        "app_name": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "environment": settings.ENVIRONMENT,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() and settings.ML_DEVICE == "cuda" else "cpu",
    }


@app.get("/", tags=["System Diagnostics"])
async def root() -> dict:
    """Welcome endpoint with API documentation references."""
    return {
        "message": "Добро пожаловать в систему «Умный конспект» API",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "api_v1": settings.API_V1_PREFIX,
    }


# Mount API V1 Master Router
app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
