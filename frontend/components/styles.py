# ### FILE: frontend/components/styles.py
"""Visual system and branded primitives for the Smart Notes Streamlit app."""

from pathlib import Path
import streamlit as st


def inject_custom_styles() -> None:
    """Read the reference-locked terminal CSS and inject it into Streamlit."""
    css_path = Path(__file__).parent.parent / "static" / "custom.css"
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def render_hero_header() -> None:
    """Render the product masthead and a compact product-intelligence hero."""
    st.markdown(
        """
        <header class="synapse-masthead">
            <div class="brand-lockup">
                <span class="brand-mark">SN</span>
                <span class="brand-name">SMART NOTES</span>
                <span class="brand-rule"></span>
                <span class="brand-context">DOCUMENT INTELLIGENCE / 01</span>
            </div>
            <div class="masthead-status"><span class="status-led"></span> API LINKED <span class="status-separator">/</span> LOCAL ENCLAVE</div>
        </header>
        <section class="synapse-hero">
            <div class="hero-copy">
                <p class="eyebrow"><span class="eyebrow-index">00</span> CAPTURE / CLEAN / STRUCTURE</p>
                <h1>Turn handwriting<br><em>into signal.</em></h1>
                <p class="hero-description">A private OCR workspace for transforming imperfect pages into searchable, editable knowledge.</p>
                <div class="hero-metrics" aria-label="System capabilities">
                    <div><strong>92</strong><span>ORTHOGRAPHIES</span></div>
                    <div><strong>100%</strong><span>LOCAL CONTROL</span></div>
                    <div><strong>∞</strong><span>REVISION MEMORY</span></div>
                </div>
            </div>
            <div class="hero-evidence">
                <div class="evidence-label"><span>LIVE EVIDENCE</span><span>PAGE 01 / RAW INPUT</span></div>
                <div class="evidence-frame">
                    <div class="evidence-grid"></div>
                    <div class="evidence-stamp">OCR<br>READY</div>
                    <div class="evidence-note">Upload a page<br>to begin the<br>recognition pass.</div>
                    <div class="evidence-corner">01</div>
                </div>
                <div class="evidence-footer"><span>PERCEPTION LAYER</span><span class="evidence-line"></span><span>WAITING FOR INPUT</span></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(status: str) -> str:
    """Return a restrained terminal status badge."""
    s = status.upper()
    if s in ("COMPLETED", "SUCCESS"):
        return f'<span class="status-badge status-success"><i></i>{s}</span>'
    if s in ("PROCESSING", "PENDING", "QUEUED"):
        return f'<span class="status-badge status-processing"><i></i>{s}</span>'
    return f'<span class="status-badge status-failed"><i></i>{s}</span>'
