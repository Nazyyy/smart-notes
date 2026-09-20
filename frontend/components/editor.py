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


def render_line_editor(
    page: Dict[str, Any], api_client: BackendAPIClient, user_id: str = "default"
) -> None:
    """Render interactive line text list with automatic NLP & LLM context corrections."""
    st.subheader(f"🤖 3. Текст с фото + Алгоритмы & LLM (Автор: {user_id})")

    lines: List[Dict[str, Any]] = page.get("lines", [])
    if not lines:
        st.info("Сегментированные строки на странице отсутствуют.")
        return

    # --------------------------------------------------------------------------
    # LLM & SEMANTIC CONTEXT CORRECTOR PANEL
    # --------------------------------------------------------------------------
    with st.expander("✨ Интеллектуальный контекстный корректор (LLM & NLP)", expanded=True):
        st.markdown(
            "Анализирует **всю страницу целиком**, восстанавливает научный контекст (физика, химия, обществознание, математика), "
            "исправляет оптические ошибки беглого почерка и сшивает переносы слов."
        )

        col_prov, col_model = st.columns([1, 1])

        provider_options = {
            "openrouter": "🌐 OpenRouter (nex-agi/nex-n2.5-pro:free) — По умолчанию",
            "heuristic": "⚡ Локальный NLP (Офлайн эвристика без LLM)",
            "ollama": "🦙 Ollama (Локально: Qwen 2.5 7B / Vikhr)",
            "openai": "✨ OpenAI (GPT-4o / GPT-4o-mini)",
            "custom": "🔧 Пользовательский OpenAI-совместимый endpoint",
        }

        with col_prov:
            prov_key = st.selectbox(
                "Провайдер интеллекта:",
                options=list(provider_options.keys()),
                format_func=lambda x: provider_options[x],
                key=f"llm_provider_{page['id']}",
            )

        with col_model:
            if prov_key == "openrouter":
                openrouter_presets = [
                    "nex-agi/nex-n2.5-pro:free",
                    "nex-agi/nex-n2.5-mini:free",
                    "deepseek/deepseek-v4-flash-0731:free",
                    "Другая модель OpenRouter...",
                ]
                or_preset_labels = {
                    "nex-agi/nex-n2.5-pro:free": "nex-agi/nex-n2.5-pro:free (100% точность, глубокий анализ / Free)",
                    "nex-agi/nex-n2.5-mini:free": "nex-agi/nex-n2.5-mini:free (Сверхбыстрая ~1.6 сек / Free)",
                    "deepseek/deepseek-v4-flash-0731:free": "deepseek/deepseek-v4-flash-0731:free (DeepSeek v4 Flash / Free)",
                    "Другая модель OpenRouter...": "Ввести имя модели OpenRouter вручную...",
                }
                sel_preset = st.selectbox(
                    "Модель OpenRouter:",
                    options=openrouter_presets,
                    format_func=lambda x: or_preset_labels.get(x, x),
                    key=f"or_preset_{page['id']}",
                )
                if sel_preset == "Другая модель OpenRouter...":
                    chosen_model = st.text_input(
                        "Имя модели:",
                        value="nex-agi/nex-n2.5-pro:free",
                        key=f"llm_model_custom_{page['id']}",
                    )
                else:
                    chosen_model = sel_preset
            elif prov_key == "ollama":
                ollama_presets = [
                    "qwen2.5:7b",
                    "rscr/vikhr_llama3.2_1b:latest",
                    "Другая локальная модель...",
                ]
                preset_labels = {
                    "qwen2.5:7b": "qwen2.5:7b (Qwen 7B — Высокая точность)",
                    "rscr/vikhr_llama3.2_1b:latest": "rscr/vikhr_llama3.2_1b (Vikhr 1B — Сверхбыстрая русская)",
                    "Другая локальная модель...": "Ввести имя другой модели вручную...",
                }
                sel_preset = st.selectbox(
                    "Модель Ollama:",
                    options=ollama_presets,
                    format_func=lambda x: preset_labels.get(x, x),
                    key=f"ollama_preset_{page['id']}",
                )
                if sel_preset == "Другая локальная модель...":
                    chosen_model = st.text_input(
                        "Имя модели:",
                        value="qwen2.5:7b",
                        key=f"llm_model_custom_{page['id']}",
                    )
                else:
                    chosen_model = sel_preset
            else:
                default_models = {
                    "openrouter": "nex-agi/nex-n2.5-pro:free",
                    "openai": "gpt-4o-mini",
                    "heuristic": "Academic-NLP-Heuristic",
                    "custom": "custom-model",
                }
                chosen_model = st.text_input(
                    "Имя модели:",
                    value=st.session_state.get("llm_model_name", default_models.get(prov_key, "nex-agi/nex-n2.5-pro:free")),
                    key=f"llm_model_input_{page['id']}",
                    disabled=(prov_key == "heuristic"),
                )
            st.session_state["llm_model_name"] = chosen_model

        col_key, col_url = st.columns([1, 1])
        if prov_key != "heuristic":
            with col_key:
                import os
                default_openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
                init_key_val = st.session_state.get(
                    "llm_api_key",
                    default_openrouter_key if prov_key == "openrouter" else "",
                )
                api_key_val = st.text_input(
                    "API Ключ:",
                    type="password",
                    value=init_key_val,
                    placeholder="sk-or-v1-..." if prov_key == "openrouter" else "sk-...",
                    key=f"llm_key_input_{page['id']}",
                )
                if api_key_val:
                    st.session_state["llm_api_key"] = api_key_val
            with col_url:
                default_urls = {
                    "openrouter": "https://openrouter.ai/api/v1",
                    "openai": "https://api.openai.com/v1",
                    "ollama": "http://localhost:11434/v1",
                    "custom": "http://localhost:8080/v1",
                }
                custom_url_val = st.text_input(
                    "Base URL:",
                    value=st.session_state.get("llm_base_url", default_urls.get(prov_key, "")),
                    key=f"llm_url_input_{page['id']}",
                )
                if custom_url_val:
                    st.session_state["llm_base_url"] = custom_url_val
        else:
            api_key_val = None
            custom_url_val = None

        col_run, col_clear = st.columns([3, 1])
        with col_run:
            if st.button("✨ Исправить ошибки контекстом всей страницы", use_container_width=True, type="primary"):
                if prov_key == "ollama":
                    sp_text = "🧠 Выполняется контекстный анализ через локальную Ollama (Qwen 7B)... Пожалуйста, подождите (~10–25 сек)..."
                elif "pro" in chosen_model:
                    sp_text = "🧠 Глубокий смысловой анализ всей страницы через nex-n2.5-pro... Занимает ~15–20 сек, запрос обрабатывается..."
                else:
                    sp_text = f"⚡ Быстрый контекстный анализ всей страницы ({chosen_model})... Занимает ~2–5 сек..."

                with st.spinner(sp_text):
                    try:
                        llm_res = api_client.correct_page_with_llm(
                            page_id=page["id"],
                            provider=prov_key,
                            api_key=api_key_val or st.session_state.get("llm_api_key"),
                            base_url=custom_url_val or st.session_state.get("llm_base_url"),
                            model=chosen_model,
                            user_id=user_id,
                        )
                        st.session_state[f"llm_res_{page['id']}"] = llm_res
                        if llm_res.get("status") == "quota_exhausted":
                            st.warning("⚠️ Дневной лимит бесплатных запросов OpenRouter исчерпан. Рекомендуется переключиться на локальный NLP или Ollama.")
                        elif llm_res.get("status") == "fallback_success":
                            st.info(f"ℹ️ {llm_res.get('provider', 'Резервная эвристика')}: найдено исправлений: {llm_res.get('corrected_count', 0)}")
                        else:
                            st.success(
                                f"Готово! Обработано строк: {llm_res.get('total_lines', 0)}, "
                                f"найдено исправлений: {llm_res.get('corrected_count', 0)}"
                            )
                    except Exception as exc:
                        st.error(f"Сбой выполнения контекстной коррекции: {exc}")

        with col_clear:
            if f"llm_res_{page['id']}" in st.session_state:
                if st.button("Скрыть предложения", use_container_width=True):
                    st.session_state.pop(f"llm_res_{page['id']}", None)
                    st.rerun()

        # Display proposed fixes and 1-click apply
        if f"llm_res_{page['id']}" in st.session_state:
            cached_res = st.session_state[f"llm_res_{page['id']}"]
            proposed_lines = cached_res.get("lines", [])
            changed_items = [l for l in proposed_lines if l.get("changed")]

            if changed_items:
                st.markdown(f"#### 🔍 Найдено **{len(changed_items)}** исправлений:")

                for item in changed_items:
                    l_idx = item.get("line_index", 0)
                    orig = item.get("original_text", "")
                    corr = item.get("corrected_text", "")
                    expl = item.get("explanation", "")

                    st.markdown(
                        f"""
                        <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 8px 12px; margin-bottom: 8px; border-radius: 4px;">
                            <span style="font-weight: 600; color: #1e40af;">Строка #{l_idx}:</span><br/>
                            <span style="color: #ef4444; text-decoration: line-through;">{orig}</span> ➔ 
                            <span style="color: #15803d; font-weight: 600;">{corr}</span><br/>
                            <span style="font-size: 0.85em; color: #64748b;">💡 {expl}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if st.button("✅ Применить все исправления к конспекту", use_container_width=True, type="primary"):
                    with st.spinner("Сохранение исправлений и адаптация профиля..."):
                        try:
                            apply_res = api_client.apply_llm_corrections(
                                page_id=page["id"],
                                corrections=proposed_lines,
                                user_id=user_id,
                            )
                            # Clear local line input caches
                            for p_line in proposed_lines:
                                line_id_k = p_line.get("line_id")
                                if line_id_k:
                                    st.session_state[f"line_input_{line_id_k}"] = p_line.get("corrected_text")
                            st.session_state.pop(f"llm_res_{page['id']}", None)
                            st.success("Все исправления успешно применены! Калибровка автора обновлена.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Не удалось применить исправления: {exc}")
            else:
                st.info("Текст распознан отлично! Оптических или контекстных ошибок не выявлено.")

    st.divider()
    st.markdown(f"Всего обнаружено строк: **{len(lines)}**")

    # --------------------------------------------------------------------------
    # LINE BY LINE CARD LIST
    # --------------------------------------------------------------------------
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

                orig_raw = line.get("original_raw_text")
                if orig_raw and orig_raw.strip() and orig_raw.strip() != new_text.strip():
                    st.caption(f"🔍 Сырой OCR до LLM-коррекции: *{orig_raw}*")

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
                                        # Send correction to adaptive personalization engine
                                        api_client.learn_personalization(
                                            original=orig_word, corrected=cand_word, user_id=user_id
                                        )
                                        st.success(f"Заменено на «{cand_word}» (калибровка '{user_id}' обновлена)!")
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
                        # Extract word modifications to calibrate author profile
                        old_tokens = text.split()
                        new_tokens = new_text.split()
                        for ot, nt in zip(old_tokens, new_tokens):
                            clean_ot = ot.strip(".,;:!?()-\"\'")
                            clean_nt = nt.strip(".,;:!?()-\"\'")
                            if clean_ot.lower() != clean_nt.lower() and len(clean_ot) >= 2 and len(clean_nt) >= 2:
                                api_client.learn_personalization(
                                    original=clean_ot, corrected=clean_nt, user_id=user_id
                                )
                        st.success(f"Сохранено (профиль '{user_id}' обновлен)!")
                    except Exception as exc:
                        st.error(f"Ошибка сохранения: {exc}")

            st.divider()

