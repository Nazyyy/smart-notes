# ### FILE: app/api/dependencies.py
"""
FastAPI Dependency Injection Providers.
Provides scoped repositories, singleton inference engines, and transaction sessions.
"""

from functools import lru_cache
from typing import AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.ml.inference import CRNNInferenceEngine
from app.repositories.document_repository import DocumentRepository
from app.repositories.page_repository import PageRepository
from app.services.storage_service import StorageService
from app.services.structurer_service import NoteStructurerService
from app.services.processing_pipeline import DocumentProcessingPipeline
from app.services.document_service import DocumentService


@lru_cache()
def get_storage_service() -> StorageService:
    """Singleton instance of local media storage service."""
    return StorageService()


@lru_cache()
def get_inference_engine():
    """Singleton instance of HTR inference engine (Hybrid Ensemble, TrOCR Transformer, or CRNN)."""
    from app.config import get_settings
    app_settings = get_settings()
    if app_settings.ML_USE_TRANSFORMER and (app_settings.ML_TRANSFORMER_PATH / "model.safetensors").exists():
        try:
            from app.ml.hybrid_ensemble import HybridEnsembleEngine
            return HybridEnsembleEngine()
        except Exception:
            try:
                from app.ml.transformer_engine import TransformerHTREngine
                return TransformerHTREngine()
            except Exception:
                pass
    return CRNNInferenceEngine()



@lru_cache()
def get_structurer_service() -> NoteStructurerService:
    """Singleton instance of NoteStructurerService."""
    return NoteStructurerService()


def get_pipeline(
    storage: StorageService = Depends(get_storage_service),
    inference_engine: CRNNInferenceEngine = Depends(get_inference_engine),
    structurer: NoteStructurerService = Depends(get_structurer_service),
) -> DocumentProcessingPipeline:
    """Factory for DocumentProcessingPipeline with injected dependencies."""
    return DocumentProcessingPipeline(
        storage_service=storage,
        inference_engine=inference_engine,
        structurer_service=structurer,
    )


def get_document_repository(
    session: AsyncSession = Depends(get_db_session),
) -> DocumentRepository:
    """Instantiate DocumentRepository bound to current transaction session."""
    return DocumentRepository(session)


def get_page_repository(
    session: AsyncSession = Depends(get_db_session),
) -> PageRepository:
    """Instantiate PageRepository bound to current transaction session."""
    return PageRepository(session)


def get_document_service(
    doc_repo: DocumentRepository = Depends(get_document_repository),
    page_repo: PageRepository = Depends(get_page_repository),
    storage: StorageService = Depends(get_storage_service),
    pipeline: DocumentProcessingPipeline = Depends(get_pipeline),
    structurer: NoteStructurerService = Depends(get_structurer_service),
) -> DocumentService:
    """Factory for high-level DocumentService."""
    return DocumentService(
        doc_repo=doc_repo,
        page_repo=page_repo,
        storage_service=storage,
        pipeline=pipeline,
        structurer=structurer,
    )
