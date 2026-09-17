# ### FILE: frontend/app.py
"""
Smart Notes (Умный конспект) - Interactive Streamlit Frontend Application.
Features drag-and-drop ingestion, OpenCV stages viewer, interactive line editor,
and KaTeX-enabled Markdown structured export.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Any, Dict, List, Optional
import streamlit as st

from frontend.api_client import BackendAPIClient
from frontend.components.styles import (
    inject_custom_styles,
    render_hero_header,
    render_status_badge,
)
from frontend.components.viewer import render_cv_stages_viewer
from frontend.components.editor import render_line_editor

# Page configuration
st.set_page_config(
    page_title="Умный конспект | AI Handwritten Notes",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom styling
inject_custom_styles()
render_hero_header()

# Initialize API client
api_client = BackendAPIClient()


# ------------------------------------------------------------------------------
# SIDEBAR: HEALTH, UPLOAD & DOCUMENT SELECTOR
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Статус и Загрузка")

    # Backend health check
    health = api_client.health_check()
    if health.get("status") == "healthy":
        device_label = f"Device: {health.get('device', 'cpu').upper()}"
        st.caption(f"🟢 Сервер API активен ({device_label})")
    else:
        st.error(f"🔴 Ошибка связи с API сервером ({health.get('error', 'unreachable')})")

    st.divider()

    # Upload Form
    with st.expander("📤 Загрузить новую страницу конспекта", expanded=False):
        with st.form("upload_form", clear_on_submit=True):
            upload_title = st.text_input("Название конспекта", placeholder="Например: Вышмат. Лекция 4")
            upload_desc = st.text_area("Описание (опционально)", placeholder="Краткое описание темы...")
            uploaded_file = st.file_uploader(
                "Фотография или скан страницы",
                type=["jpg", "jpeg", "png", "webp"],
            )
            submit_upload = st.form_submit_button("🚀 Загрузить и распознать")

            if submit_upload:
                if not upload_title:
                    st.warning("Пожалуйста, укажите название документа.")
                elif uploaded_file is None:
                    st.warning("Пожалуйста, выберите файл изображения.")
                else:
                    with st.spinner("Загрузка и запуск нейросетевого конвейера..."):
                        try:
                            file_bytes = uploaded_file.getvalue()
                            doc_res = api_client.upload_document(
                                title=upload_title,
                                file_bytes=file_bytes,
                                filename=uploaded_file.name,
                                description=upload_desc,
                                async_background=False,
                            )
                            st.success(f"Документ '{upload_title}' успешно обработан!")
                            st.session_state["selected_doc_id"] = doc_res["id"]
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Сбой загрузки или распознавания: {exc}")

    st.divider()

    # Document Selection List
    st.subheader("📚 Мои конспекты")
    docs_resp = api_client.list_documents()
    docs_list = docs_resp.get("items", [])

    if not docs_list:
        st.info("Пока нет сохраненных конспектов. Загрузите первую страницу выше!")
        selected_doc_id = None
    else:
        doc_options = {d["id"]: f"{d['title']} ({d['status']})" for d in docs_list}
        doc_ids = list(doc_options.keys())

        # Select first doc by default if not set
        if "selected_doc_id" not in st.session_state or st.session_state["selected_doc_id"] not in doc_ids:
            st.session_state["selected_doc_id"] = doc_ids[0]

        selected_doc_id = st.selectbox(
            "Выберите документ:",
            options=doc_ids,
            format_func=lambda x: doc_options[x],
            index=doc_ids.index(st.session_state["selected_doc_id"]),
            key="doc_selector",
        )
        st.session_state["selected_doc_id"] = selected_doc_id

        if st.button("🗑️ Удалить выбранный документ", use_container_width=True):
            try:
                api_client.delete_document(selected_doc_id)
                st.success("Документ удален.")
                st.session_state.pop("selected_doc_id", None)
                st.rerun()
            except Exception as exc:
                st.error(f"Не удалось удалить документ: {exc}")


# ------------------------------------------------------------------------------
# MAIN WORKSPACE
# ------------------------------------------------------------------------------
if not selected_doc_id:
    st.markdown(
        """
        ### Добро пожаловать в «Умный конспект»!
        Для начала работы загрузите фотографию конспекта в панели слева.
        
        **Возможности системы:**
        1. **Компьютерное зрение:** автоматическое выравнивание перспективы, устранение наклона, фильтрация неравномерных теней от смартфона.
        2. **Точная сегментация:** нарезка рукописных строк на основе проекционных профилей (HPP).
        3. **Нейросетевое распознавание:** глубокая архитектура CRNN (CNN + BiGRU + CTC) для кириллицы, латиницы и математических символов.
        4. **Интеллектуальное структурирование:** распознавание списков, формул в нотации KaTeX/LaTeX и экспорт в чистый Markdown.
        """
    )
else:
    try:
        current_doc = api_client.get_document(selected_doc_id)
    except Exception as exc:
        st.error(f"Не удалось загрузить данные документа: {exc}")
        st.stop()

    pages = current_doc.get("pages", [])
    primary_page = pages[0] if pages else None

    # Header Card
    col_info, col_actions = st.columns([7, 5])

    with col_info:
        st.markdown(f"## {current_doc.get('title')}")
        if current_doc.get("description"):
            st.caption(current_doc.get("description"))

        status_html = render_status_badge(current_doc.get("status", "PENDING"))
        st.markdown(f"Статус: {status_html}", unsafe_allow_html=True)

    with col_actions:
        st.write("")
        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if primary_page and st.button("🔄 Перераспознать", use_container_width=True):
                with st.spinner("Перезапуск конвейера..."):
                    try:
                        api_client.reprocess_page(primary_page["id"])
                        st.success("Перераспознавание запущено!")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Ошибка: {exc}")

        with col_btn2:
            if st.button("🔄 Обновить вид", use_container_width=True):
                st.rerun()

    st.divider()

    # Workspace Tabs
    tab_cv, tab_editor, tab_markdown = st.tabs([
        "🖼️ CV Конвейер & Сегментация",
        "✏️ Построчный редактор",
        "📑 Структурированный Markdown & Экспорт",
    ])

    with tab_cv:
        if primary_page:
            render_cv_stages_viewer(primary_page)
        else:
            st.info("Страницы для данного документа не найдены.")

    with tab_editor:
        if primary_page:
            render_line_editor(primary_page, api_client)
        else:
            st.info("Страницы для редактирования отсутствуют.")

    with tab_markdown:
        st.subheader("📑 Структурированный конспект")

        try:
            markdown_content = api_client.download_export(current_doc["id"], export_format="MARKDOWN")
        except Exception:
            markdown_content = f"# {current_doc['title']}\n\n*(Экспорт генерируется...)*\n"

        # Action Buttons for Downloads
        col_dl_md, col_dl_txt, col_regen = st.columns([3, 3, 4])

        with col_dl_md:
            st.download_button(
                label="📥 Скачать Markdown (.md)",
                data=markdown_content,
                file_name=f"{current_doc['title']}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with col_dl_txt:
            try:
                txt_content = api_client.download_export(current_doc["id"], export_format="TXT")
            except Exception:
                txt_content = markdown_content

            st.download_button(
                label="📄 Скачать Текст (.txt)",
                data=txt_content,
                file_name=f"{current_doc['title']}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with col_regen:
            if st.button("✨ Перегенерировать структуру", use_container_width=True):
                try:
                    api_client.export_document(current_doc["id"], export_format="MARKDOWN")
                    st.success("Структура успешно обновлена!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Ошибка генерации: {exc}")

        st.markdown("### Предварительный просмотр")
        st.markdown(markdown_content)

        with st.expander("Исходный код Markdown"):
            st.code(markdown_content, language="markdown")
