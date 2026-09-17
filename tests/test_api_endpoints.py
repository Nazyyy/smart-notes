# ### FILE: tests/test_api_endpoints.py
"""
Integration Tests for FastAPI REST Endpoints.
Verifies health probes, multipart document uploads, error wrapping, line edits,
page debug artifacts, recognition triggering, status polling, exports, and document deletion.
"""

import io
from uuid import uuid4
import cv2
import numpy as np
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(async_client: AsyncClient):
    """Verify GET /health returns 200 and healthy status."""
    resp = await async_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "device" in data


@pytest.mark.asyncio
async def test_root_endpoint(async_client: AsyncClient):
    """Verify GET / returns welcome payload with documentation links."""
    resp = await async_client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "docs_url" in data
    assert data["api_v1"] == "/api/v1"


@pytest.mark.asyncio
async def test_upload_document_success(async_client: AsyncClient, synthetic_page_image: np.ndarray):
    """Verify POST /api/v1/documents/upload processes handwritten note."""
    # Encode synthetic image to JPEG bytes
    success, buffer = cv2.imencode(".jpg", synthetic_page_image)
    assert success
    file_bytes = buffer.tobytes()

    files = {"file": ("lecture_page.jpg", file_bytes, "image/jpeg")}
    data = {
        "title": "Тестовая лекция 1",
        "description": "Описание тестового документа",
        "process_immediately": "true",
        "async_background": "false",
    }

    resp = await async_client.post("/api/v1/documents/upload", data=data, files=files)
    assert resp.status_code == 201
    doc = resp.json()

    assert doc["title"] == "Тестовая лекция 1"
    assert doc["status"] in ("COMPLETED", "PROCESSING")
    assert len(doc["pages"]) >= 1

    page = doc["pages"][0]
    assert page["page_number"] == 1
    assert len(page["lines"]) >= 3


@pytest.mark.asyncio
async def test_upload_invalid_extension(async_client: AsyncClient):
    """Verify uploading an unpermitted file extension returns 400 Bad Request."""
    fake_file = io.BytesIO(b"malicious executable payload")
    files = {"file": ("exploit.exe", fake_file, "application/octet-stream")}
    data = {"title": "Exploit Attempt"}

    resp = await async_client.post("/api/v1/documents/upload", data=data, files=files)
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"] == "InvalidFilePayloadException"


@pytest.mark.asyncio
async def test_get_document_not_found(async_client: AsyncClient):
    """Verify fetching non-existent UUID returns 404."""
    random_id = str(uuid4())
    resp = await async_client.get(f"/api/v1/documents/{random_id}")
    assert resp.status_code == 404
    data = resp.json()
    assert data["error"] == "DocumentNotFoundException"


@pytest.mark.asyncio
async def test_list_documents(async_client: AsyncClient):
    """Verify GET /api/v1/documents returns list response format."""
    resp = await async_client.get("/api/v1/documents?limit=10&offset=0")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "total_count" in data
    assert isinstance(data["items"], list)


@pytest.mark.asyncio
async def test_update_line_transcription_and_export(async_client: AsyncClient, synthetic_page_image: np.ndarray):
    """Upload document, edit recognized line transcription, and verify export reflects changes."""
    success, buffer = cv2.imencode(".jpg", synthetic_page_image)
    files = {"file": ("math_notes.jpg", buffer.tobytes(), "image/jpeg")}
    data = {"title": "Конспект по матанализу", "async_background": "false"}

    upload_resp = await async_client.post("/api/v1/documents/upload", data=data, files=files)
    assert upload_resp.status_code == 201
    doc = upload_resp.json()

    doc_id = doc["id"]
    page = doc["pages"][0]
    page_id = page["id"]
    first_line = page["lines"][0]
    line_id = first_line["id"]

    # Update line text manually
    custom_text = "Исправленный вручную заголовок"
    update_resp = await async_client.put(
        f"/api/v1/pages/{page_id}/lines/{line_id}",
        json={"recognized_text": custom_text, "confidence": 1.0},
    )
    assert update_resp.status_code == 200
    updated_line = update_resp.json()
    assert updated_line["recognized_text"] == custom_text

    # Generate Markdown export
    export_resp = await async_client.post(
        f"/api/v1/documents/{doc_id}/export",
        json={"export_format": "MARKDOWN"},
    )
    assert export_resp.status_code == 200
    export_data = export_resp.json()
    assert custom_text in export_data["content"]

    # Download export file
    download_resp = await async_client.get(f"/api/v1/documents/{doc_id}/export/MARKDOWN/download")
    assert download_resp.status_code == 200
    assert custom_text in download_resp.text


@pytest.mark.asyncio
async def test_page_details_and_artifacts(async_client: AsyncClient, synthetic_page_image: np.ndarray):
    """Verify fetching page details and intermediate CV debug artifacts."""
    success, buffer = cv2.imencode(".jpg", synthetic_page_image)
    files = {"file": ("page_details_test.jpg", buffer.tobytes(), "image/jpeg")}
    data = {"title": "Страница для проверки артефактов", "async_background": "false"}

    upload_resp = await async_client.post("/api/v1/documents/upload", data=data, files=files)
    assert upload_resp.status_code == 201
    doc = upload_resp.json()
    page_id = doc["pages"][0]["id"]

    # 1. Fetch valid page details
    page_resp = await async_client.get(f"/api/v1/pages/{page_id}")
    assert page_resp.status_code == 200
    page_data = page_resp.json()
    assert page_data["id"] == page_id
    assert len(page_data["lines"]) >= 3

    # 2. Fetch page debug artifacts
    artifacts_resp = await async_client.get(f"/api/v1/pages/{page_id}/debug-artifacts")
    assert artifacts_resp.status_code == 200
    artifacts = artifacts_resp.json()
    assert "raw_image_url" in artifacts
    assert "deskewed_image_url" in artifacts
    assert "shadow_removed_url" in artifacts
    assert "binarized_url" in artifacts
    assert "projection_profile_url" in artifacts
    assert "segmented_lines_overlay_url" in artifacts

    # 3. Non-existent page ID should return 404
    missing_id = str(uuid4())
    missing_page_resp = await async_client.get(f"/api/v1/pages/{missing_id}")
    assert missing_page_resp.status_code == 404
    missing_art_resp = await async_client.get(f"/api/v1/pages/{missing_id}/debug-artifacts")
    assert missing_art_resp.status_code == 404


@pytest.mark.asyncio
async def test_recognition_status_and_reprocess(async_client: AsyncClient, synthetic_page_image: np.ndarray):
    """Verify checking recognition progress status and triggering page reprocessing."""
    success, buffer = cv2.imencode(".jpg", synthetic_page_image)
    files = {"file": ("reprocess_test.jpg", buffer.tobytes(), "image/jpeg")}
    data = {"title": "Документ для повторного распознавания", "async_background": "false"}

    upload_resp = await async_client.post("/api/v1/documents/upload", data=data, files=files)
    assert upload_resp.status_code == 201
    doc = upload_resp.json()
    doc_id = doc["id"]
    page_id = doc["pages"][0]["id"]

    # 1. Check recognition status
    status_resp = await async_client.get(f"/api/v1/recognition/documents/{doc_id}/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["document_id"] == doc_id
    assert status_data["status"] == "COMPLETED"
    assert status_data["estimated_progress_pct"] == 100.0

    # 2. Trigger page reprocessing
    reprocess_resp = await async_client.post(
        f"/api/v1/recognition/pages/{page_id}/reprocess",
        json={"beam_width": 3, "use_beam_search": True},
    )
    assert reprocess_resp.status_code == 200
    rep_data = reprocess_resp.json()
    assert rep_data["status"] == "QUEUED"
    assert rep_data["page_id"] == page_id

    # 3. Reprocess non-existent page ID should return error
    missing_id = str(uuid4())
    missing_rep = await async_client.post(f"/api/v1/recognition/pages/{missing_id}/reprocess", json={})
    assert missing_rep.status_code in (400, 404, 500)


@pytest.mark.asyncio
async def test_export_txt_and_latex_formats(async_client: AsyncClient, synthetic_page_image: np.ndarray):
    """Verify exporting document to TXT and LATEX formats and checking download responses."""
    success, buffer = cv2.imencode(".jpg", synthetic_page_image)
    files = {"file": ("export_formats.jpg", buffer.tobytes(), "image/jpeg")}
    data = {"title": "Тест различных форматов экспорта", "async_background": "false"}

    upload_resp = await async_client.post("/api/v1/documents/upload", data=data, files=files)
    assert upload_resp.status_code == 201
    doc = upload_resp.json()
    doc_id = doc["id"]

    # 1. Plain text export
    txt_resp = await async_client.post(
        f"/api/v1/documents/{doc_id}/export",
        json={"export_format": "TXT"},
    )
    assert txt_resp.status_code == 200
    txt_data = txt_resp.json()
    assert txt_data["export_format"] == "TXT"
    assert len(txt_data["content"]) > 0

    txt_down = await async_client.get(f"/api/v1/documents/{doc_id}/export/TXT/download")
    assert txt_down.status_code == 200
    assert len(txt_down.text) > 0

    # 2. LaTeX export
    latex_resp = await async_client.post(
        f"/api/v1/documents/{doc_id}/export",
        json={"export_format": "LATEX"},
    )
    assert latex_resp.status_code == 200
    latex_data = latex_resp.json()
    assert latex_data["export_format"] == "LATEX"
    assert "\\begin{document}" in latex_data["content"]

    latex_down = await async_client.get(f"/api/v1/documents/{doc_id}/export/LATEX/download")
    assert latex_down.status_code == 200
    assert "\\documentclass" in latex_down.text

    # 3. Export non-existent document
    missing_id = str(uuid4())
    missing_export = await async_client.post(
        f"/api/v1/documents/{missing_id}/export",
        json={"export_format": "TXT"},
    )
    assert missing_export.status_code == 404

    missing_down = await async_client.get(f"/api/v1/documents/{missing_id}/export/TXT/download")
    assert missing_down.status_code == 404


@pytest.mark.asyncio
async def test_delete_document_and_cascade(async_client: AsyncClient, synthetic_page_image: np.ndarray):
    """Verify deleting a document removes it from storage and database."""
    success, buffer = cv2.imencode(".jpg", synthetic_page_image)
    files = {"file": ("doc_to_delete.jpg", buffer.tobytes(), "image/jpeg")}
    data = {"title": "Документ для удаления", "async_background": "false"}

    upload_resp = await async_client.post("/api/v1/documents/upload", data=data, files=files)
    assert upload_resp.status_code == 201
    doc = upload_resp.json()
    doc_id = doc["id"]

    # 1. Delete the document
    del_resp = await async_client.delete(f"/api/v1/documents/{doc_id}")
    assert del_resp.status_code == 204

    # 2. Fetch deleted document should return 404
    get_resp = await async_client.get(f"/api/v1/documents/{doc_id}")
    assert get_resp.status_code == 404

    # 3. Deleting non-existent document should return 404
    missing_id = str(uuid4())
    del_missing = await async_client.delete(f"/api/v1/documents/{missing_id}")
    assert del_missing.status_code == 404


@pytest.mark.asyncio
async def test_page_endpoints_and_transcription_edit(async_client: AsyncClient, synthetic_page_image: np.ndarray):
    """Verify page retrieval, debug artifacts URLs, and line transcription updates."""
    success, buffer = cv2.imencode(".jpg", synthetic_page_image)
    files = {"file": ("page_test.jpg", buffer.tobytes(), "image/jpeg")}
    data = {"title": "Страничный тест", "async_background": "false"}

    upload_resp = await async_client.post("/api/v1/documents/upload", data=data, files=files)
    assert upload_resp.status_code == 201
    doc = upload_resp.json()
    assert len(doc["pages"]) > 0
    page = doc["pages"][0]
    page_id = page["id"]

    # 1. Get page details
    page_resp = await async_client.get(f"/api/v1/pages/{page_id}")
    assert page_resp.status_code == 200
    page_data = page_resp.json()
    assert page_data["id"] == page_id
    assert len(page_data["lines"]) > 0

    # 2. Get missing page details
    missing_id = str(uuid4())
    missing_page_resp = await async_client.get(f"/api/v1/pages/{missing_id}")
    assert missing_page_resp.status_code == 404

    # 3. Get debug artifacts URLs
    debug_resp = await async_client.get(f"/api/v1/pages/{page_id}/debug-artifacts")
    assert debug_resp.status_code == 200
    debug_data = debug_resp.json()
    assert "deskewed_image_url" in debug_data
    assert "binarized_url" in debug_data

    # 4. Get debug artifacts for non-existent page
    missing_debug = await async_client.get(f"/api/v1/pages/{missing_id}/debug-artifacts")
    assert missing_debug.status_code == 404

    # 5. Update transcription of an existing line
    first_line = page_data["lines"][0]
    line_id = first_line["id"]
    new_text = "Исправленная строка конспекта ручным вводом"
    edit_resp = await async_client.put(
        f"/api/v1/pages/{page_id}/lines/{line_id}",
        json={"recognized_text": new_text, "confidence": 0.99},
    )
    assert edit_resp.status_code == 200
    updated_line = edit_resp.json()
    assert updated_line["recognized_text"] == new_text
    assert updated_line["confidence"] == 0.99

    # 6. Update transcription with non-existent line ID
    missing_line_resp = await async_client.put(
        f"/api/v1/pages/{page_id}/lines/{missing_id}",
        json={"recognized_text": "text", "confidence": 0.5},
    )
    assert missing_line_resp.status_code == 404

    # 7. Reprocess valid page
    reprocess_resp = await async_client.post(f"/api/v1/recognition/pages/{page_id}/reprocess", json={})
    assert reprocess_resp.status_code == 200
    rep_data = reprocess_resp.json()
    assert rep_data["status"] == "QUEUED"
    assert rep_data["page_id"] == page_id

    # 8. Recognition status for document
    status_resp = await async_client.get(f"/api/v1/recognition/documents/{doc['id']}/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert "estimated_progress_pct" in status_data
    assert status_data["document_id"] == doc["id"]

    # 9. List documents and download raw source
    list_resp = await async_client.get("/api/v1/documents?skip=0&limit=10")
    assert list_resp.status_code == 200
    doc_list = list_resp.json()
    assert doc_list["total_count"] >= 1
    assert len(doc_list["items"]) >= 1

    download_resp = await async_client.get(f"/api/v1/documents/{doc['id']}/export/MARKDOWN/download")
    assert download_resp.status_code == 200
    assert len(download_resp.content) > 0

    # 10. Test word suggestions endpoint
    sug_resp = await async_client.get("/api/v1/recognition/suggest?word=биосоииальное&top_k=3")
    assert sug_resp.status_code == 200
    sug_list = sug_resp.json()
    assert isinstance(sug_list, list)
    assert len(sug_list) >= 1
    assert any("биосоциальное" in c["word"] for c in sug_list)

    # 11. Test line suggestions endpoint
    line_sug_resp = await async_client.post(
        "/api/v1/recognition/suggest-line",
        json={"text": "человек биосоииальное существо", "confidence": 0.75, "top_k": 3},
    )
    assert line_sug_resp.status_code == 200
    line_sugs = line_sug_resp.json()
    assert isinstance(line_sugs, dict)
    assert "биосоииальное" in line_sugs
    assert any("биосоциальное" in c["word"] for c in line_sugs["биосоииальное"])


