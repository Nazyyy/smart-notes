# ### FILE: frontend/api_client.py
"""
HTTP API Client for interacting with the Smart Notes FastAPI Backend.
Handles serialization, multi-part streaming uploads, error wrapping, and timeouts.
"""

import os
from typing import Any, Dict, List, Optional
import requests

DEFAULT_BACKEND_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000/api/v1")


class BackendAPIClient:
    """Client for synchronous communication between Streamlit UI and FastAPI Backend."""

    def __init__(self, base_url: str = DEFAULT_BACKEND_URL, timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """Verify connection to backend server."""
        try:
            # Note health check is mounted on root, not /api/v1
            root_url = self.base_url.replace("/api/v1", "")
            resp = self.session.get(f"{root_url}/health", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            return {"status": "unreachable", "error": str(exc)}

    def upload_document(
        self,
        title: str,
        file_bytes: bytes,
        filename: str,
        description: Optional[str] = None,
        async_background: bool = False,
    ) -> Dict[str, Any]:
        """Upload image and trigger processing pipeline."""
        url = f"{self.base_url}/documents/upload"
        files = {"file": (filename, file_bytes, "image/jpeg")}
        data = {
            "title": title,
            "description": description or "",
            "process_immediately": "true",
            "async_background": "true" if async_background else "false",
        }
        resp = self.session.post(url, data=data, files=files, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def list_documents(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Fetch list of uploaded documents."""
        url = f"{self.base_url}/documents"
        params = {"limit": limit, "offset": offset}
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_document(self, document_id: str) -> Dict[str, Any]:
        """Fetch document details with pages and lines."""
        url = f"{self.base_url}/documents/{document_id}"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def delete_document(self, document_id: str) -> bool:
        """Delete document by ID."""
        url = f"{self.base_url}/documents/{document_id}"
        resp = self.session.delete(url, timeout=self.timeout)
        return resp.status_code == 204

    def get_page_debug_artifacts(self, page_id: str) -> Dict[str, Any]:
        """Fetch relative artifact paths for intermediate CV stages."""
        url = f"{self.base_url}/pages/{page_id}/debug-artifacts"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def update_line_text(self, page_id: str, line_id: str, new_text: str) -> Dict[str, Any]:
        """Update transcription text for a specific line."""
        url = f"{self.base_url}/pages/{page_id}/lines/{line_id}"
        payload = {"recognized_text": new_text}
        resp = self.session.put(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def reprocess_page(self, page_id: str, beam_width: int = 5) -> Dict[str, Any]:
        """Trigger reprocessing of a single page."""
        url = f"{self.base_url}/recognition/pages/{page_id}/reprocess"
        payload = {"beam_width": beam_width}
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def export_document(self, document_id: str, export_format: str = "MARKDOWN") -> Dict[str, Any]:
        """Generate structured text export."""
        url = f"{self.base_url}/documents/{document_id}/export"
        payload = {"export_format": export_format}
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def download_export(self, document_id: str, export_format: str = "MARKDOWN") -> str:
        """Download raw export text content."""
        url = f"{self.base_url}/documents/{document_id}/export/{export_format}/download"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.text
