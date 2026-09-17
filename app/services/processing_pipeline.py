# ### FILE: app/services/processing_pipeline.py
"""
End-to-End Handwritten Notes Recognition Processing Pipeline.
Orchestrates CV rectification, shadow removal, HPP segmentation, CRNN inference,
and database persistence.
"""

from pathlib import Path
from typing import Optional, List, Tuple
from uuid import UUID
import cv2
import numpy as np

from app.config import get_settings
from app.core.exceptions import ImageProcessingException, ModelInferenceException
from app.core.logging import get_logger
from app.cv import (
    rectify_document_geometry,
    suppress_shadows_and_denoise,
    adaptive_binarize,
    segment_text_lines,
    save_pipeline_debug_artifacts,
)
from app.ml.inference import CRNNInferenceEngine
from app.models.entities import Document, Page, TextLine, DocumentStatus, PageStatus, ExportFormat
from app.repositories.document_repository import DocumentRepository
from app.repositories.page_repository import PageRepository
from app.services.storage_service import StorageService
from app.services.structurer_service import NoteStructurerService

logger = get_logger(__name__)
settings = get_settings()


class DocumentProcessingPipeline:
    """Production coordinator executing end-to-end CV and ML pipeline on document pages."""

    def __init__(
        self,
        storage_service: StorageService,
        inference_engine: Optional[object] = None,
        structurer_service: Optional[NoteStructurerService] = None,
    ) -> None:
        self.storage_service = storage_service
        if inference_engine is not None:
            self.inference_engine = inference_engine
        else:
            if settings.ML_USE_TRANSFORMER and (settings.ML_TRANSFORMER_PATH / "model.safetensors").exists():
                try:
                    from app.ml.transformer_engine import TransformerHTREngine
                    self.inference_engine = TransformerHTREngine()
                except Exception as exc:
                    logger.warning("TrOCR init failed in pipeline, fallback to CRNN: %s", exc)
                    self.inference_engine = CRNNInferenceEngine()
            else:
                self.inference_engine = CRNNInferenceEngine()
        self.structurer = structurer_service or NoteStructurerService()


    async def execute_page_pipeline(
        self,
        document_id: UUID,
        page_id: UUID,
        doc_repo: DocumentRepository,
        page_repo: PageRepository,
        beam_width: int = 5,
    ) -> Page:
        """
        Execute full lifecycle pipeline on a page:
        1. Read raw image from disk.
        2. Rectify perspective and deskew.
        3. Equalize lighting and suppress shadows.
        4. Binarize ink foreground.
        5. Extract text line bounding boxes via HPP.
        6. Predict text via PyTorch CRNN.
        7. Save debug artifacts and persist line entities.
        8. Auto-generate structured Markdown export.
        """
        logger.info("Starting processing pipeline for Document %s, Page %s", document_id, page_id)

        # Update document & page status to PROCESSING
        await doc_repo.update_status(document_id, DocumentStatus.PROCESSING)
        await page_repo.update(page_id, {"status": PageStatus.PENDING.value})

        page = await page_repo.get_by_id(page_id)
        if not page:
            raise ImageProcessingException("pipeline_init", f"Page with ID {page_id} not found.")

        raw_path = Path(page.raw_image_path)
        if not raw_path.exists():
            error_msg = f"Raw image missing on disk: {raw_path}"
            await page_repo.update(page_id, {"status": PageStatus.FAILED.value})
            await doc_repo.update_status(document_id, DocumentStatus.FAILED, error_message=error_msg)
            raise ImageProcessingException("load_image", error_msg)

        try:
            # 1. Load image
            raw_bgr = cv2.imread(str(raw_path))
            if raw_bgr is None or raw_bgr.size == 0:
                raise ImageProcessingException("load_image", f"Corrupt or unreadable image file: {raw_path}")

            orig_h, orig_w = raw_bgr.shape[:2]

            # Downscale if image exceeds max dimension to avoid memory exhaustion
            if max(orig_h, orig_w) > settings.MAX_IMAGE_DIMENSION:
                scale = settings.MAX_IMAGE_DIMENSION / float(max(orig_h, orig_w))
                new_w, new_h = int(orig_w * scale), int(orig_h * scale)
                raw_bgr = cv2.resize(raw_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 2. Rectify perspective and deskew
            rectified_bgr, skew_angle = rectify_document_geometry(raw_bgr)
            rect_h, rect_w = rectified_bgr.shape[:2]

            # 3. Suppress shadows and normalize lighting
            shadow_suppressed = suppress_shadows_and_denoise(rectified_bgr)
            gray = cv2.cvtColor(shadow_suppressed, cv2.COLOR_BGR2GRAY)

            # 4. Adaptive binarization
            binary_mask = adaptive_binarize(gray)

            # 5. Line segmentation via HPP
            bboxes, crops, hpp = segment_text_lines(rectified_bgr, binary_mask)

            # 6. ML Recognition (CRNN + CTC)
            transcriptions: List[str] = []
            confidences: List[float] = []

            if crops:
                # Use beam search if configured
                predictions = self.inference_engine.predict_batch(
                    crops, use_beam_search=(beam_width > 1)
                )
                for text, conf in predictions:
                    transcriptions.append(text)
                    confidences.append(conf)

            # 7. Persist debug artifacts
            debug_dir = self.storage_service.get_page_debug_directory(document_id, page_id)
            save_pipeline_debug_artifacts(
                debug_dir=debug_dir,
                raw_image=raw_bgr,
                rectified_image=rectified_bgr,
                shadow_suppressed=shadow_suppressed,
                binary_image=binary_mask,
                hpp=hpp,
                bboxes=bboxes,
                crops=crops,
                transcriptions=transcriptions,
                confidences=confidences,
            )

            # 8. Create and persist TextLine database records
            line_entities: List[TextLine] = []
            for idx, (x, y, w, h) in enumerate(bboxes):
                text = transcriptions[idx] if idx < len(transcriptions) else ""
                conf = confidences[idx] if idx < len(confidences) else 0.0
                crop_rel_path = str(debug_dir / "lines" / f"line_{idx:03d}.jpg")

                line_entities.append(
                    TextLine(
                        page_id=page_id,
                        line_index=idx,
                        bbox_x=x,
                        bbox_y=y,
                        bbox_w=w,
                        bbox_h=h,
                        cropped_image_path=crop_rel_path,
                        recognized_text=text,
                        confidence=conf,
                    )
                )

            saved_lines = await page_repo.replace_page_lines(page_id, line_entities)

            # Update Page record with dimensions and artifacts
            await page_repo.update(
                page_id,
                {
                    "width": rect_w,
                    "height": rect_h,
                    "skew_angle": skew_angle,
                    "processed_image_path": str(debug_dir / "04_binarized.png"),
                    "deskewed_image_path": str(debug_dir / "02_rectified.jpg"),
                    "debug_dir_path": str(debug_dir),
                    "status": PageStatus.COMPLETED.value,
                },
            )

            # 9. Auto-generate structured Markdown export
            doc = await doc_repo.get_by_id(document_id)
            doc_title = doc.title if doc else "Конспект"
            markdown_content = self.structurer.structure_lines_to_markdown(saved_lines, doc_title)

            export_dir = self.storage_service.get_export_directory(document_id)
            export_file_path = export_dir / "notes.md"
            await self.storage_service.write_text_file(export_file_path, markdown_content)

            await doc_repo.save_export(
                document_id=document_id,
                export_format=ExportFormat.MARKDOWN.value,
                file_path=str(export_file_path),
                content=markdown_content,
            )

            # Mark document as COMPLETED
            await doc_repo.update_status(document_id, DocumentStatus.COMPLETED)
            logger.info("Successfully completed pipeline for Document %s, Page %s", document_id, page_id)

            refreshed_page = await page_repo.get_page_with_lines(page_id)
            return refreshed_page or page

        except Exception as exc:
            logger.error("Processing pipeline failed for Document %s, Page %s: %s", document_id, page_id, exc, exc_info=True)
            await page_repo.update(page_id, {"status": PageStatus.FAILED.value})
            await doc_repo.update_status(document_id, DocumentStatus.FAILED, error_message=str(exc))
            raise
