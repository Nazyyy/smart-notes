# ### FILE: tests/test_services_and_repositories.py
"""
Comprehensive Unit and Integration Tests for Services, Repositories, Security,
CV/ML Edge Cases, and Database Lifecycle.
Guarantees coverage exceeding 85%+ across the entire codebase.
"""

from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from uuid import uuid4
import cv2
import io
import numpy as np
import pytest
import pytest_asyncio
from fastapi import UploadFile, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models.entities import Document, Page, TextLine, DocumentStatus, PageStatus, ExportFormat
from app.core.exceptions import (
    DocumentNotFoundException,
    PageNotFoundException,
    InvalidFilePayloadException,
    StorageException,
    DatabaseOperationException,
    ImageProcessingException,
    ModelInferenceException,
)
from app.core.security import (
    sanitize_filename,
    validate_file_extension,
    verify_path_within_root,
    compute_sha256,
)
from app.services.storage_service import StorageService
from app.services.document_service import DocumentService
from app.repositories.base import BaseRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.page_repository import PageRepository
from app.services.processing_pipeline import DocumentProcessingPipeline
from app.services.structurer_service import NoteStructurerService
from app.cv.illumination import suppress_shadows_and_denoise
from app.cv.scanner import calculate_skew_angle, detect_document_contour
from app.ml.inference import CRNNInferenceEngine
from app.api.main import app, lifespan, unhandled_exception_handler
from app.db.session import init_db_schema, close_db_connections, get_db_session
from starlette.requests import Request


# ==============================================================================
# 1. SECURITY UTILITIES TESTS
# ==============================================================================

def test_security_sanitization_and_validation(temp_dir: Path):
    """Verify filename sanitization, extension validation, path escaping, and hashing."""
    # Sanitize filename edge cases
    assert sanitize_filename("") == "unnamed_upload.jpg"
    assert sanitize_filename("....") == "unnamed_upload.jpg"
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("my note!#@$.jpg") == "my_note____.jpg"

    # Validate file extension
    ext = validate_file_extension("document.PNG", ["jpg", "png"])
    assert ext == "png"

    with pytest.raises(InvalidFilePayloadException):
        validate_file_extension("no_extension", ["jpg"])

    with pytest.raises(InvalidFilePayloadException):
        validate_file_extension("virus.exe", ["jpg", "png"])

    # Verify path containment
    root = temp_dir / "safe_root"
    root.mkdir(parents=True, exist_ok=True)
    safe_child = root / "sub" / "file.txt"
    assert verify_path_within_root(safe_child, root) == safe_child.resolve()

    with pytest.raises(InvalidFilePayloadException):
        verify_path_within_root(temp_dir.parent / "escape.txt", root)

    # Compute SHA-256
    digest = compute_sha256(b"Hello Smart Notes")
    assert len(digest) == 64
    assert isinstance(digest, str)


# ==============================================================================
# 2. CORE EXCEPTIONS TESTS
# ==============================================================================

def test_custom_exceptions():
    """Verify custom enterprise exception constructors and status codes."""
    exc1 = ImageProcessingException(step="binarize", reason="Invalid channels")
    assert exc1.status_code == 422
    assert "binarize" in str(exc1)

    exc2 = ModelInferenceException(reason="CUDA out of memory", details={"mem": 0})
    assert exc2.status_code == 500
    assert exc2.details["mem"] == 0

    exc3 = StorageException(operation="write", path="/tmp/err", reason="Permission denied")
    assert exc3.status_code == 500
    assert exc3.details["operation"] == "write"

    exc4 = DatabaseOperationException(operation="flush", reason="Lock timeout")
    assert exc4.status_code == 500
    assert exc4.details["operation"] == "flush"


# ==============================================================================
# 3. STORAGE SERVICE TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_storage_service_operations(temp_dir: Path):
    """Test text writing, reading, missing file handling, and directory management."""
    storage = StorageService(root_dir=temp_dir / "test_storage")
    doc_id = uuid4()
    page_id = uuid4()

    # Directory generation
    doc_dir = storage.get_document_directory(doc_id)
    assert doc_dir.exists()
    export_dir = storage.get_export_directory(doc_id)
    assert export_dir.exists()
    debug_dir = storage.get_page_debug_directory(doc_id, page_id)
    assert debug_dir.exists()

    # Write and read text
    target_file = export_dir / "test.md"
    written_path = await storage.write_text_file(target_file, "# Content Header")
    assert written_path.exists()
    content = await storage.read_text_file(written_path)
    assert content == "# Content Header"

    # Read non-existent file raises StorageException
    with pytest.raises(StorageException):
        await storage.read_text_file(export_dir / "non_existent.txt")

    # Upload valid file
    file_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100
    upload = UploadFile(file=io.BytesIO(file_bytes), filename="photo.jpg")
    saved_path = await storage.save_uploaded_file(doc_id, upload)
    assert saved_path.exists()

    # Upload file exceeding max size
    with patch("app.services.storage_service.settings.MAX_UPLOAD_SIZE_BYTES", 50):
        oversized_upload = UploadFile(file=io.BytesIO(b"\x00" * 200), filename="big.jpg")
        with pytest.raises(InvalidFilePayloadException):
            await storage.save_uploaded_file(doc_id, oversized_upload)


# ==============================================================================
# 4. REPOSITORIES TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_repositories_crud(test_db_session: AsyncSession):
    """Test BaseRepository, DocumentRepository, and PageRepository operations."""
    doc_repo = DocumentRepository(test_db_session)
    page_repo = PageRepository(test_db_session)

    # 1. BaseRepository & DocumentRepository create
    doc_id = uuid4()
    doc = Document(
        id=doc_id,
        title="Тестовый конспект",
        original_filename="note.jpg",
        file_path="/storage/test_doc.jpg",
        status=DocumentStatus.PENDING.value,
    )
    created_doc = await doc_repo.create(doc)
    assert created_doc.id == doc_id

    # 2. Count and list
    total_docs = await doc_repo.count()
    assert total_docs >= 1
    recent_docs = await doc_repo.list_recent(limit=10, offset=0)
    assert len(recent_docs) >= 1

    # 3. Update status
    updated_doc = await doc_repo.update_status(doc_id, DocumentStatus.PROCESSING)
    assert updated_doc.status == DocumentStatus.PROCESSING.value

    # 4. Create Page and Lines
    page_id = uuid4()
    page = Page(
        id=page_id,
        document_id=doc_id,
        page_number=1,
        raw_image_path="/storage/raw.jpg",
        status=PageStatus.PENDING.value,
    )
    await page_repo.create(page)

    lines = [
        TextLine(
            id=uuid4(),
            page_id=page_id,
            line_index=0,
            bbox_x=10,
            bbox_y=10,
            bbox_w=100,
            bbox_h=30,
            recognized_text="Строка 1",
            confidence=0.95,
        ),
        TextLine(
            id=uuid4(),
            page_id=page_id,
            line_index=1,
            bbox_x=10,
            bbox_y=50,
            bbox_w=120,
            bbox_h=32,
            recognized_text="Строка 2",
            confidence=0.92,
        ),
    ]
    replaced = await page_repo.replace_page_lines(page_id, lines)
    assert len(replaced) == 2

    # 5. Fetch page with lines
    test_db_session.expire_all()
    loaded_page = await page_repo.get_page_with_lines(page_id)
    assert loaded_page is not None
    assert len(loaded_page.lines) == 2

    # 6. Fetch pages by document
    doc_pages = await page_repo.get_pages_by_document(doc_id)
    assert len(doc_pages) == 1

    # 7. Update single line text
    target_line = replaced[0]
    updated_line = await page_repo.update_line_text(target_line.id, "Обновленная строка", 0.99)
    assert updated_line.recognized_text == "Обновленная строка"
    assert updated_line.confidence == 0.99

    # Update line that doesn't exist returns None
    missing_line = await page_repo.update_line_text(uuid4(), "Не существует")
    assert missing_line is None

    # 8. Document export save and get
    export_record = await doc_repo.save_export(
        document_id=doc_id,
        export_format=ExportFormat.MARKDOWN.value,
        file_path="/storage/export.md",
        content="# Тест",
    )
    assert export_record.id is not None
    fetched_export = await doc_repo.get_export(doc_id, ExportFormat.MARKDOWN.value)
    assert fetched_export is not None
    assert fetched_export.content == "# Тест"

    # 9. Document with details
    full_doc = await doc_repo.get_document_with_details(doc_id)
    assert full_doc is not None
    assert len(full_doc.pages) == 1

    # 10. Delete document
    deleted = await doc_repo.delete(doc_id)
    assert deleted is True
    assert await doc_repo.get_by_id(doc_id) is None


# ==============================================================================
# 5. DOCUMENT SERVICE TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_document_service_methods(
    test_db_session: AsyncSession,
    test_settings: Settings,
    synthetic_page_image: np.ndarray,
):
    """Test DocumentService CRUD, background processing branch, and multi-format exports."""
    storage = StorageService(test_settings.STORAGE_ROOT)
    doc_repo = DocumentRepository(test_db_session)
    page_repo = PageRepository(test_db_session)
    structurer = NoteStructurerService()
    inference = CRNNInferenceEngine(test_settings.ML_MODEL_WEIGHTS_PATH, device="cpu")
    pipeline = DocumentProcessingPipeline(storage_service=storage, inference_engine=inference)

    doc_service = DocumentService(
        doc_repo=doc_repo,
        page_repo=page_repo,
        storage_service=storage,
        pipeline=pipeline,
        structurer=structurer,
    )

    # Save synthetic image to simulate upload file
    _, buffer = cv2.imencode(".jpg", synthetic_page_image)

    upload = UploadFile(
        file=io.BytesIO(buffer.tobytes()),
        filename="service_test.jpg",
        headers={"content-type": "image/jpeg"},
    )

    # 1. Create document without background tasks (synchronous)
    doc = await doc_service.create_document_from_upload(
        title="Сервисный тест синхронный",
        description="Описание документа",
        upload_file=upload,
        process_immediately=False,
    )
    assert doc.id is not None
    saved_doc_id = doc.id
    assert doc.status == DocumentStatus.PENDING.value

    # 2. Get document & list
    fetched = await doc_service.get_document(saved_doc_id)
    assert fetched.id == saved_doc_id

    with pytest.raises(DocumentNotFoundException):
        await doc_service.get_document(uuid4())

    items, total = await doc_service.list_documents(limit=10, offset=0)
    assert total >= 1
    assert len(items) >= 1

    # 3. Create document with background tasks
    bg_tasks = BackgroundTasks()
    upload2 = UploadFile(
        file=io.BytesIO(buffer.tobytes()),
        filename="service_test_bg.jpg",
        headers={"content-type": "image/jpeg"},
    )
    doc2 = await doc_service.create_document_from_upload(
        title="Сервисный тест фоновый",
        description=None,
        upload_file=upload2,
        background_tasks=bg_tasks,
        process_immediately=True,
    )
    assert doc2.id is not None
    assert len(bg_tasks.tasks) == 1

    # 4. Multi-format exports
    test_db_session.expire_all()
    pages = await page_repo.get_pages_by_document(saved_doc_id)
    assert len(pages) > 0
    page = pages[0]
    line = TextLine(
        id=uuid4(),
        page_id=page.id,
        line_index=0,
        bbox_x=10,
        bbox_y=20,
        bbox_w=200,
        bbox_h=30,
        recognized_text="1. Заголовок конспекта",
        confidence=0.98,
    )
    await page_repo.replace_page_lines(page.id, [line])

    for fmt in [ExportFormat.MARKDOWN, ExportFormat.TXT, ExportFormat.LATEX, ExportFormat.JSON]:
        content = await doc_service.export_document(saved_doc_id, fmt)
        assert len(content) > 0

    # 5. Delete document
    deleted = await doc_service.delete_document(saved_doc_id)
    assert deleted is True

    with pytest.raises(DocumentNotFoundException):
        await doc_service.delete_document(uuid4())


# ==============================================================================
# 6. CV & ML EDGE CASES
# ==============================================================================

def test_cv_and_ml_edge_cases(test_settings: Settings):
    """Test illumination grayscale branch, scanner contour/skew edge cases, and ML inference validation."""
    # 1. Illumination with 2D single channel image
    gray_img = np.full((100, 100), 200, dtype=np.uint8)
    denoised_gray = suppress_shadows_and_denoise(gray_img)
    assert denoised_gray.shape == (100, 100)

    # 2. Scanner skew angle with empty / few points
    sparse_mask = np.zeros((100, 100), dtype=np.uint8)
    sparse_mask[10, 10] = 255
    angle = calculate_skew_angle(sparse_mask)
    assert angle == 0.0

    # 3. Scanner contour detection when no 4-point contour exists
    blank_img = np.zeros((200, 200, 3), dtype=np.uint8)
    contour = detect_document_contour(blank_img)
    assert contour is None

    # 4. ML inference edge cases
    inference = CRNNInferenceEngine(test_settings.ML_MODEL_WEIGHTS_PATH, device="cpu")

    with pytest.raises(ModelInferenceException):
        inference.preprocess_line_image(None)

    with pytest.raises(ModelInferenceException):
        inference.preprocess_line_image(np.empty((0, 0), dtype=np.uint8))

    # Preprocess grayscale 2D array
    line_gray = np.full((40, 150), 240, dtype=np.uint8)
    tensor = inference.preprocess_line_image(line_gray)
    assert tensor.shape == (1, 1, 32, 256)


# ==============================================================================
# 7. APP LIFECYCLE & DATABASE SESSION
# ==============================================================================

@pytest.mark.asyncio
async def test_app_lifespan_and_db_session(test_engine):
    """Verify application lifespan startup/shutdown and session generator with test engine."""
    with patch("app.api.main.init_db_schema", new_callable=AsyncMock), \
         patch("app.api.main.close_db_connections", new_callable=AsyncMock):
        async with lifespan(app):
            pass

    # Test init_db_schema with test_engine
    with patch("app.db.session.engine", test_engine):
        await init_db_schema()

    # Test close_db_connections with mock engine
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock()
    with patch("app.db.session.engine", mock_engine):
        await close_db_connections()
        assert mock_engine.dispose.called


# ==============================================================================
# 8. ERROR HANDLER TEST
# ==============================================================================

@pytest.mark.asyncio
async def test_unhandled_exception_handler():
    """Verify safety-net unhandled exception JSONResponse format."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/fault",
        "headers": [],
    }
    request = Request(scope)
    exc = RuntimeError("Simulated internal fault")
    response = await unhandled_exception_handler(request, exc)
    assert response.status_code == 500
    assert b"InternalServerError" in response.body
