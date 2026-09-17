# ### FILE: frontend/components/editor.py
"""
Interactive Line-by-Line Transcription Editor Component.
Allows inspecting line crops, visual highlighting of uncertain words,
and one-click substitution of high-probability optical cursive candidates.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import streamlit as st
from PIL import Image
from frontend.api_client import BackendAPIClient


def render_confidence_indicator(confidence: float) -> str:
    """Format confidence score into colored HTML tag."""
    pct = int(confidence * 100)
    if confidence >= 0.80:
        color = "#10b981"
        status = "высокая"
    elif confidence >= 0.55:
        color = "#f59e0b"
        status = "средняя"
    else:
        color = "#ef4444"
        status = "требует проверки"
    return f'<span style="color: {color}; font-weight: 600; font-size: 0.9em;">● {pct}% уверенность ({status})</span>'


def render_highlighted_preview(text: str, uncertain_words: List[str]) -> str:
    """Render text with low-confidence / ambiguous words highlighted in gold marker."""
    if not text:
        return ""
    words = text.split()
    html_parts = []
    uncertain_set = set(w.lower().strip(".,;:!?()-\"\'") for w in uncertain_words)

    for w in words:
        clean = w.lower().strip(".,;:!?()-\"\'")
        if clean in uncertain_set:
            html_parts.append(
                f'<span style="background-color: #fef08a; color: #854d0e; padding: 2px 6px; '
                f'border-radius: 4px; font-weight: 600; border: 1px dashed #eab308;" title="Потенциальная оптическая ошибка">{w}</span>'
            )
        else:
            html_parts.append(f"<span>{w}</span>")

    return " ".join(html_parts)


def render_line_editor(page: Dict[str, Any], api_client: BackendAPIClient) -> None:
    """Render interactive card-based list of lines for review and correction."""
    st.subheader("✏️ Построчный редактор с матрицей подсказок")

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

        # Key for this line input
        input_key = f"line_input_{line_id}"
        if input_key not in st.session_state:
            st.session_state[input_key] = text

        current_val = st.session_state[input_key]

        # Identify uncertain words and calculate optical alternatives via backend API
        cache_key = f"cands_{line_id}_{hash(current_val)}"
        if cache_key in st.session_state:
            uncertain_words_with_cands = st.session_state[cache_key]
        else:
            uncertain_words_with_cands = api_client.get_line_suggestions(
                text=current_val, confidence=conf, top_k=3
            )
            st.session_state[cache_key] = uncertain_words_with_cands


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

                if uncertain_words_with_cands:
                    highlighted_html = render_highlighted_preview(current_val, list(uncertain_words_with_cands.keys()))
                    st.markdown(
                        f'<div style="margin-bottom: 6px; font-size: 0.95em;">{highlighted_html}</div>',
                        unsafe_allow_html=True,
                    )

                new_text = st.text_input(
                    label=f"Текст строки #{line_idx}",
                    value=st.session_state[input_key],
                    key=input_key,
                    label_visibility="collapsed",
                )

                # Render one-click suggestion chips for uncertain words
                if uncertain_words_with_cands:
                    st.caption("💡 Быстрые подсказки матрицы почерка:")
                    for orig_word, cands in uncertain_words_with_cands.items():
                        c_cols = st.columns(len(cands))
                        for c_idx, cand_info in enumerate(cands):
                            cand_word = cand_info["word"]
                            with c_cols[c_idx]:
                                btn_label = f'"{orig_word}" ➔ {cand_word}'
                                if st.button(btn_label, key=f"btn_sug_{line_id}_{orig_word}_{c_idx}"):
                                    # Substitute candidate word
                                    pattern = re.compile(re.escape(orig_word), re.IGNORECASE)
                                    updated = pattern.sub(cand_word, st.session_state[input_key], count=1)
                                    st.session_state[input_key] = updated
                                    try:
                                        api_client.update_line_text(
                                            page_id=page["id"],
                                            line_id=line_id,
                                            new_text=updated,
                                        )
                                        st.success(f"Заменено на «{cand_word}»!")
                                        st.rerun()
                                    except Exception as exc:
                                        st.error(f"Ошибка: {exc}")

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
