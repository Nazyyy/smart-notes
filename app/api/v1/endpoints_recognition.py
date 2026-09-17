# ### FILE: app/api/v1/endpoints_recognition.py
"""
Recognition Pipeline Trigger and Status Monitoring Endpoints.
"""

from uuid import UUID
from fastapi import APIRouter, Depends, BackgroundTasks, Body
from app.api.dependencies import (
    get_document_service,
    get_pipeline,
    get_document_repository,
    get_page_repository,
)
from app.core.exceptions import PageNotFoundException
from app.models.entities import DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.repositories.page_repository import PageRepository
from app.schemas.recognition import (
    LineSuggestionRequest,
    RecognitionProgressResponse,
    RecognitionTriggerRequest,
)
from app.services.document_service import DocumentService
from app.services.processing_pipeline import DocumentProcessingPipeline

router = APIRouter(prefix="/recognition", tags=["Recognition"])


@router.post(
    "/pages/{page_id}/reprocess",
    summary="Trigger reprocessing of a specific page",
)
async def reprocess_page(
    page_id: UUID,
    background_tasks: BackgroundTasks,
    request_params: RecognitionTriggerRequest = Body(default_factory=RecognitionTriggerRequest),
    pipeline: DocumentProcessingPipeline = Depends(get_pipeline),
    doc_repo: DocumentRepository = Depends(get_document_repository),
    page_repo: PageRepository = Depends(get_page_repository),
) -> dict:
    """Manually re-run the optical rectification, segmentation, and recognition pipeline."""
    page = await page_repo.get_by_id(page_id)
    if not page:
        raise PageNotFoundException(page_id)

    background_tasks.add_task(
        pipeline.execute_page_pipeline,
        document_id=page.document_id,
        page_id=page.id,
        doc_repo=doc_repo,
        page_repo=page_repo,
        beam_width=request_params.beam_width or 5,
    )

    return {
        "status": "QUEUED",
        "message": f"Reprocessing scheduled for page {page_id}.",
        "document_id": str(page.document_id),
        "page_id": str(page.id),
    }


@router.get(
    "/documents/{document_id}/status",
    response_model=RecognitionProgressResponse,
    summary="Get processing status and line metrics for document",
)
async def get_recognition_status(
    document_id: UUID,
    doc_service: DocumentService = Depends(get_document_service),
) -> RecognitionProgressResponse:
    """Check processing lifecycle state and line metrics for progress polling."""
    doc = await doc_service.get_document(document_id)

    total_lines = 0
    recognized_lines = 0

    for page in doc.pages:
        total_lines += len(page.lines)
        recognized_lines += sum(1 for line in page.lines if line.recognized_text)

    if doc.status == DocumentStatus.COMPLETED.value:
        progress = 100.0
    elif doc.status == DocumentStatus.PROCESSING.value:
        progress = 50.0 if total_lines == 0 else (recognized_lines / float(total_lines)) * 100.0
    elif doc.status == DocumentStatus.FAILED.value:
        progress = 0.0
    else:
        progress = 10.0

    return RecognitionProgressResponse(
        document_id=doc.id,
        status=doc.status,
        processed_lines=recognized_lines,
        total_lines=total_lines,
        estimated_progress_pct=min(100.0, max(0.0, progress)),
    )


@router.get(
    "/suggest",
    summary="Get optical handwriting candidates and alternatives for a word",
)
async def get_word_suggestions(
    word: str,
    context: str = "",
    top_k: int = 3,
) -> list[dict]:
    """Return top handwriting confusion alternatives for uncertain or misrecognized words."""
    from app.ml.handwriting_confusion import get_handwriting_confusion_corrector
    corrector = get_handwriting_confusion_corrector()
    ctx_words = context.split() if context else None
    return corrector.get_word_candidates(word, context_words=ctx_words, top_k=top_k)


@router.post(
    "/suggest-line",
    summary="Get optical handwriting candidates for all uncertain words in a line",
)
async def suggest_line(req: LineSuggestionRequest) -> dict[str, list[dict]]:
    """Scan line text, identify low-confidence / non-lexicon words, and return ranked candidates."""
    from app.ml.handwriting_confusion import get_handwriting_confusion_corrector
    corrector = get_handwriting_confusion_corrector()
    words = [w.strip(".,;:!?()-\"\'") for w in req.text.split() if len(w.strip(".,;:!?()-\"\'")) >= 3]
    result: dict[str, list[dict]] = {}
    for w in words:
        clean_lower = w.lower()
        if clean_lower not in corrector.vocabulary or req.confidence < 0.88:
            cands = corrector.get_word_candidates(w, context_words=words, top_k=req.top_k)
            valid_cands = [c for c in cands if c["word"].lower() != clean_lower]
            if valid_cands:
                result[w] = valid_cands
    return result


from pydantic import BaseModel, Field


class PersonalizationLearnRequest(BaseModel):
    original: str = Field(..., description="Original OCR transcribed word")
    corrected: str = Field(..., description="User's intended/corrected word")
    user_id: str = Field("default", description="Identifier of the writer")


@router.post(
    "/personalization/learn",
    summary="Record user correction to personalize optical handwriting weights",
)
async def learn_personalization(req: PersonalizationLearnRequest) -> dict:
    """Record manual user edit and calibrate the optical handwriting confusion matrix."""
    from app.ml.personalization import get_user_profile
    from app.ml.handwriting_confusion import get_handwriting_confusion_corrector

    profile = get_user_profile(req.user_id)
    learned = profile.record_correction(req.original, req.corrected)
    corrector = get_handwriting_confusion_corrector()
    profile.apply_to_confusion_corrector(corrector)

    return {
        "status": "learned",
        "learned_pairs": learned,
        "stats": profile.get_stats(),
    }


@router.get(
    "/personalization/profile",
    summary="Get user handwriting calibration statistics and learned confusions",
)
async def get_personalization_profile(user_id: str = "default") -> dict:
    """Return metrics, top confused letter pairs, and custom words for user profile."""
    from app.ml.personalization import get_user_profile
    profile = get_user_profile(user_id)
    return profile.get_stats()


@router.post(
    "/personalization/reset",
    summary="Reset user handwriting calibration profile",
)
async def reset_personalization_profile(user_id: str = "default") -> dict:
    """Reset all learned personal handwriting substitutions to factory defaults."""
    from app.ml.personalization import get_user_profile
    from app.ml.handwriting_confusion import get_handwriting_confusion_corrector
    profile = get_user_profile(user_id)
    profile.reset()
    corrector = get_handwriting_confusion_corrector()
    corrector._build_confusion_weights()
    return {"status": "reset", "user_id": user_id}


