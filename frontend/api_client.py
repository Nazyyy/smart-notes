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

    def __init__(self, base_url: str = DEFAULT_BACKEND_URL, timeout: int = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """Verify connection to backend server."""
        try:
            # Note health check is mounted on root, not /api/v1
            root_url = self.base_url.replace("/api/v1", "")
            resp = self.session.get(f"{root_url}/health", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            return {"status": "busy", "message": "Сервер выполняет нейросетевую обработку..."}
        except Exception as exc:
            return {"status": "unreachable", "error": str(exc)}

    def upload_document(
        self,
        title: str,
        file_bytes: bytes,
        filename: str,
        author: str = "default",
        description: Optional[str] = None,
        async_background: bool = False,
    ) -> Dict[str, Any]:
        """Upload image and trigger processing pipeline."""
        url = f"{self.base_url}/documents/upload"
        files = {"file": (filename, file_bytes, "image/jpeg")}
        data = {
            "title": title,
            "author": author,
            "description": description or "",
            "process_immediately": "true",
            "async_background": "true" if async_background else "false",
        }
        resp = self.session.post(url, data=data, files=files, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def list_documents(
        self, limit: int = 50, offset: int = 0, author: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch list of uploaded documents."""
        url = f"{self.base_url}/documents"
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if author:
            params["author"] = author
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

    def synthesize_ai_study_guide(
        self,
        document_id: str,
        provider: str = "openrouter",
        model: str = "nex-agi/nex-n2.5-pro:free",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        length_mode: str = "medium",
        enrich_facts: bool = False,
        creativity_mode: str = "strict",
    ) -> Dict[str, Any]:
        """Synthesize deep, beautifully structured academic study guide using LLM with customization."""
        url = f"{self.base_url}/documents/{document_id}/export/ai-synthesize"
        payload = {
            "export_format": "MARKDOWN",
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "length_mode": length_mode,
            "enrich_facts": enrich_facts,
            "creativity_mode": creativity_mode,
        }
        resp = self.session.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()

    def get_interactive_kit(
        self,
        document_id: str,
        provider: str = "openrouter",
        model: str = "nex-agi/nex-n2.5-pro:free",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate or retrieve interactive study kit (flashcards, cloze tests, quiz)."""
        url = f"{self.base_url}/documents/{document_id}/interactive-kit"
        payload = {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
        }
        resp = self.session.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()


    def download_export(self, document_id: str, export_format: str = "MARKDOWN") -> str:
        """Download raw export text content."""
        url = f"{self.base_url}/documents/{document_id}/export/{export_format}/download"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.text


    def get_word_suggestions(
        self, word: str, context: Optional[str] = None, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Fetch optical cursive word suggestions from backend."""
        try:
            url = f"{self.base_url}/recognition/suggest"
            params: Dict[str, Any] = {"word": word, "top_k": top_k}
            if context:
                params["context"] = context
            resp = self.session.get(url, params=params, timeout=3)
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception:
            return []

    def get_line_suggestions(
        self, text: str, confidence: float = 1.0, top_k: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch optical cursive suggestions for all uncertain words in a line."""
        try:
            url = f"{self.base_url}/recognition/suggest-line"
            payload = {"text": text, "confidence": confidence, "top_k": top_k}
            resp = self.session.post(url, json=payload, timeout=3)
            if resp.status_code == 200:
                return resp.json()
            return {}
        except Exception:
            return {}

    def learn_personalization(
        self, original: str, corrected: str, user_id: str = "default"
    ) -> Dict[str, Any]:
        """Send manual correction to backend to calibrate user handwriting weights."""
        try:
            url = f"{self.base_url}/recognition/personalization/learn"
            payload = {"original": original, "corrected": corrected, "user_id": user_id}
            resp = self.session.post(url, json=payload, timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return {}
        except Exception:
            return {}

    def get_personalization_profile(self, user_id: str = "default") -> Dict[str, Any]:
        """Fetch user handwriting profile metrics and learned substitutions."""
        try:
            url = f"{self.base_url}/recognition/personalization/profile"
            resp = self.session.get(url, params={"user_id": user_id}, timeout=3)
            if resp.status_code == 200:
                return resp.json()
            return {}
        except Exception:
            return {}

    def reset_personalization_profile(self, user_id: str = "default") -> bool:
        """Reset learned user handwriting profile."""
        try:
            url = f"{self.base_url}/recognition/personalization/reset"
            resp = self.session.post(url, params={"user_id": user_id}, timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def list_users(self) -> List[Dict[str, Any]]:
        """Fetch list of all handwriting profiles."""
        try:
            url = f"{self.base_url}/recognition/personalization/users"
            resp = self.session.get(url, timeout=3)
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception:
            return []

    def create_user(self, user_name: str, display_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a new handwriting user profile."""
        url = f"{self.base_url}/recognition/personalization/users"
        payload = {"user_name": user_name, "display_name": display_name}
        resp = self.session.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json()

    def correct_page_with_llm(
        self,
        page_id: str,
        provider: str = "openrouter",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "nex-agi/nex-n2.5-pro:free",
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Trigger full-page contextual correction using LLM or offline NLP."""
        url = f"{self.base_url}/recognition/llm/correct"
        payload = {
            "page_id": page_id,
            "provider": provider,
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "user_id": user_id,
        }
        resp = self.session.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()

    def apply_llm_corrections(
        self,
        page_id: str,
        corrections: List[Dict[str, Any]],
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Commit LLM corrections to database and calibration profile."""
        url = f"{self.base_url}/recognition/llm/apply"
        payload = {
            "page_id": page_id,
            "user_id": user_id,
            "corrections": corrections,
        }
        resp = self.session.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()




