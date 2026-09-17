# ### FILE: frontend/components/editor.py
"""
Interactive Line-by-Line Transcription Editor Component.
Allows inspecting line crops and correcting OCR transcriptions.
"""

from pathlib import Path
from typing import Any, Dict, List
import streamlit as st
from PIL import Image
from frontend.api_client import BackendAPIClient


def render_confidence_indicator(confidence: float) -> str:
    """Format confidence score into colored HTML tag."""
    pct = int(confidence * 100)
    if confidence >= 0.80:
        color = "#10b981"
    elif confidence >= 0.50:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    return f'<span style="color: {color}; font-weight: 600;">{pct}% уверенность</span>'


def render_line_editor(page: Dict[str, Any], api_client: BackendAPIClient) -> None:
    """Render interactive card-based list of lines for review and correction."""
    st.subheader("✏️ Построчный редактор распознавания")

    lines: List[Dict[str, Any]] = page.get("lines", [])
    if not lines:
        st.info("Сегментированные строки на странице отсутствуют.")
        return

    st.markdown(f"Всего обнаружено строк: **{len(lines)}**")

    for line in lines:
        line_id = line.get("id")
        line_idx = line.get("line_index", 0)
        text = line.get("recognized_text", "")
        conf = float(line.get("confidence", 0.0))
        crop_path_str = line.get("cropped_image_path")

        with st.container():
            col_crop, col_edit, col_btn = st.columns([4, 6, 2])

            with col_crop:
                if crop_path_str and Path(crop_path_str).exists():
                    st.image(
                        Image.open(crop_path_str),
                        caption=f"Строка #{line_idx}",
                        use_column_width=True,
                    )
                else:
                    st.caption(f"Строка #{line_idx} [изображение недоступно]")

            with col_edit:
                st.markdown(render_confidence_indicator(conf), unsafe_allow_html=True)
                new_text = st.text_input(
                    label=f"Текст строки #{line_idx}",
                    value=text,
                    key=f"line_input_{line_id}",
                    label_visibility="collapsed",
                )

            with col_btn:
                st.write("")  # Vertical alignment spacer
                if st.button("💾 Сохранить", key=f"btn_save_{line_id}"):
                    try:
                        api_client.update_line_text(
                            page_id=page["id"],
                            line_id=line_id,
                            new_text=new_text,
                        )
                        st.success("Сохранено!")
                    except Exception as exc:
                        st.error(f"Ошибка сохранения: {exc}")

            st.divider()
