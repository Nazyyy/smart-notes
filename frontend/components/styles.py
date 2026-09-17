# ### FILE: frontend/components/styles.py
"""
CSS Injection and Branding Components for Streamlit.
"""

from pathlib import Path
import streamlit as st


def inject_custom_styles() -> None:
    """Read static custom.css and inject it directly into the Streamlit app head."""
    css_path = Path(__file__).parent.parent / "static" / "custom.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def render_hero_header() -> None:
    """Render the master branding header."""
    st.markdown(
        """
        <div class="smart-notes-hero">
            <div class="smart-notes-title">
                <span>📝 Умный конспект</span>
            </div>
            <div class="smart-notes-subtitle">
                Интеллектуальная система распознавания, очистки и структурирования рукописных конспектов
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(status: str) -> str:
    """Return HTML markup for colorful status badge."""
    s = status.upper()
    if s in ("COMPLETED", "SUCCESS"):
        return f'<span class="badge-completed">✓ {s}</span>'
    elif s in ("PROCESSING", "PENDING", "QUEUED"):
        return f'<span class="badge-processing">⏳ {s}</span>'
    else:
        return f'<span class="badge-failed">✗ {s}</span>'
