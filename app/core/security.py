# ### FILE: app/core/security.py
"""
Security Utilities: Path Traversal Prevention, Sanitization, and Validation.
Implements defensive measures against OWASP Top 10 vulnerabilities.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Tuple
from app.core.exceptions import InvalidFilePayloadException


def sanitize_filename(filename: str) -> str:
    """
    Sanitize an uploaded filename to eliminate Path Traversal attacks,
    null bytes, and shell metacharacters.
    """
    if not filename:
        return "unnamed_upload.jpg"

    # Remove path directory separators (POSIX & Windows)
    basename = os.path.basename(filename.replace("\\", "/"))

    # Remove dangerous characters, allow only alphanumeric, underscores, hyphens, and dots
    sanitized = re.sub(r"[^A-Za-z0-9_.\-]", "_", basename)

    # Prevent hidden files or relative path navigations
    sanitized = sanitized.lstrip(".")
    if not sanitized:
        sanitized = "unnamed_upload.jpg"

    return sanitized


def validate_file_extension(filename: str, allowed_extensions: list[str]) -> str:
    """
    Validate that the file's extension matches the allowed list.
    Returns normalized lowercase extension without leading dot.
    """
    if "." not in filename:
        raise InvalidFilePayloadException(
            reason=f"Uploaded file '{filename}' has no extension.",
            details={"filename": filename},
        )

    ext = filename.rsplit(".", 1)[1].lower().strip()
    clean_allowed = [e.lower().lstrip(".") for e in allowed_extensions]

    if ext not in clean_allowed:
        raise InvalidFilePayloadException(
            reason=f"Extension '.{ext}' is not permitted. Allowed: {clean_allowed}",
            details={"extension": ext, "allowed": clean_allowed},
        )

    return ext


def verify_path_within_root(target_path: Path, root_directory: Path) -> Path:
    """
    Ensure the resolved target path is strictly contained within the intended root directory,
    preventing arbitrary file reads or writes via symlink / relative traversal.
    """
    resolved_target = target_path.resolve()
    resolved_root = root_directory.resolve()

    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise InvalidFilePayloadException(
            reason="Access violation: Target path escapes storage boundary.",
            details={"target": str(resolved_target), "root": str(resolved_root)},
        ) from exc

    return resolved_target


def compute_sha256(data: bytes) -> str:
    """Compute deterministic SHA-256 digest of binary payload."""
    return hashlib.sha256(data).hexdigest()
