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
from frontend.components.viewer import (
    render_photo_transformations,
    render_line_segmentation_stage,
)
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
    elif health.get("status") == "busy":
        st.info("⏳ Сервер выполняет ресурсоемкое распознавание конспекта...")
    else:
        st.error(f"🔴 Ошибка связи с API сервером ({health.get('error', 'unreachable')})")

    st.divider()

    # --------------------------------------------------------------------------
    # USER SELECTOR & PROFILES
    # --------------------------------------------------------------------------
    st.subheader("👤 Выбор пользователя")
    users_list = api_client.list_users()
    if not users_list:
        users_list = [{"user_id": "default", "display_name": "Основной пользователь", "total_corrections": 0}]

    user_map = {
        u["user_id"]: f"{u.get('display_name') or u['user_id']} ({u.get('total_corrections', 0)} правок)"
        for u in users_list
    }
    user_ids = list(user_map.keys())

    if "active_user_id" not in st.session_state or st.session_state["active_user_id"] not in user_ids:
        st.session_state["active_user_id"] = user_ids[0]

    active_user_id = st.selectbox(
        "Текущий автор конспекта:",
        options=user_ids,
        format_func=lambda x: user_map.get(x, x),
        index=user_ids.index(st.session_state["active_user_id"]),
        key="active_user_selector",
    )
    st.session_state["active_user_id"] = active_user_id

    # Create new profile
    with st.expander("➕ Добавить нового автора", expanded=False):
        with st.form("new_user_form", clear_on_submit=True):
            new_u_name = st.text_input("Имя / Псевдоним автора", placeholder="Например: Дмитрий, Студент 2")
            if st.form_submit_button("Создать профиль"):
                if new_u_name.strip():
                    created = api_client.create_user(user_name=new_u_name.strip(), display_name=new_u_name.strip())
                    st.session_state["active_user_id"] = created.get("user_id", new_u_name.strip())
                    st.success(f"Профиль '{new_u_name}' успешно создан!")
                    st.rerun()
                else:
                    st.warning("Введите имя автора.")

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
                    with st.spinner("⚡ Обработка изображения и высокоскоростное нейросетевое распознавание..."):
                        try:
                            file_bytes = uploaded_file.getvalue()
                            doc_res = api_client.upload_document(
                                title=upload_title,
                                file_bytes=file_bytes,
                                filename=uploaded_file.name,
                                author=active_user_id,
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
    filter_by_user = st.checkbox("Только конспекты выбранного автора", value=False)
    docs_resp = api_client.list_documents(author=active_user_id if filter_by_user else None)
    docs_list = docs_resp.get("items", [])

    if not docs_list:
        st.info("Пока нет сохраненных конспектов. Загрузите первую страницу выше!")
        selected_doc_id = None
    else:
        doc_options = {d["id"]: f"{d['title']} ({d['status']}) [{d.get('author', 'default')}]" for d in docs_list}
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

    st.divider()
    st.subheader(f"🎯 Калибровка автора ({active_user_id})")
    calib_enabled = st.checkbox("Адаптация под почерк автора", value=True)
    if calib_enabled:
        profile_stats = api_client.get_personalization_profile(user_id=active_user_id)
        if profile_stats:
            total_c = profile_stats.get("total_corrections", 0)
            learned_sub = profile_stats.get("learned_substitutions_count", 0)
            top_pairs = profile_stats.get("top_confusions", [])
            st.caption(f"✍️ Выучено правок: **{total_c}** | Замен букв: **{learned_sub}**")
            if top_pairs:
                pairs_str = ", ".join([f"{p['pair']} ({p['count']}x)" for p in top_pairs[:4]])
                st.caption(f"Особенности почерка: `{pairs_str}`")
        if st.button("🔄 Сбросить калибровку этого автора", use_container_width=True):
            api_client.reset_personalization_profile(user_id=active_user_id)
            st.toast(f"Калибровка для '{active_user_id}' сброшена.")
            st.rerun()



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

    # Workspace Tabs: 4 distinct stages
    tab_photo, tab_segmentation, tab_editor, tab_markdown = st.tabs([
        "📸 1. Фото и преобразования",
        "📐 2. Разметка строк",
        "🤖 3. Текст + Алгоритмы & LLM",
        "📝 4. Итоговый результат",
    ])

    with tab_photo:
        if primary_page:
            render_photo_transformations(primary_page)
        else:
            st.info("Страницы для данного документа не найдены.")

    with tab_segmentation:
        if primary_page:
            render_line_segmentation_stage(primary_page)
        else:
            st.info("Данные сегментации отсутствуют.")

    with tab_editor:
        if primary_page:
            render_line_editor(primary_page, api_client, user_id=active_user_id)
        else:
            st.info("Страницы для распознавания отсутствуют.")

    with tab_markdown:
        st.subheader("📝 4. Итоговый результат и Интерактивное обучение")
        st.caption("Глубокий академический конспект, интерактивные карточки терминов и тренажер активного вспоминания.")

        # ----------------------------------------------------------------------
        # CUSTOMIZATION & GENERATION CONTROLS
        # ----------------------------------------------------------------------
        with st.expander("⚙️ Настройки кастомизации конспекта (Объем, факты, стиль)", expanded=False):
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                length_choice = st.selectbox(
                    "📏 Длина конспекта:",
                    options=["Средний (стандарт)", "Короткий (шпаргалка)", "Подробный (исчерпывающий)"],
                    index=0,
                    help="Выбирает степень сжатия и детализации материала",
                )
                length_mode_map = {
                    "Средний (стандарт)": "medium",
                    "Короткий (шпаргалка)": "short",
                    "Подробный (исчерпывающий)": "detailed",
                }
            with col_c2:
                creativity_choice = st.selectbox(
                    "🎨 Стиль изложения («отсебятина»):",
                    options=["Строгий (только факты лекции)", "Сбалансированный (понятный научный)", "Креативный (жизненные аналогии)"],
                    index=0,
                    help="Регулирует, насколько ИИ может использовать понятные аналогии и пояснения простыми словами",
                )
                creativity_mode_map = {
                    "Строгий (только факты лекции)": "strict",
                    "Сбалансированный (понятный научный)": "balanced",
                    "Креативный (жизненные аналогии)": "creative",
                }
            with col_c3:
                enrich_facts = st.checkbox(
                    "📚 Добавить расширенные факты",
                    value=False,
                    help="ИИ добавит историко-научные справки, интересные контекстные факты и имена",
                )

        col_ai_btn, col_algo_btn = st.columns([6, 4])
        with col_ai_btn:
            if st.button("✨ Синтезировать конспект через ИИ (nex-n2.5-pro)", type="primary", use_container_width=True):
                with st.spinner("🤖 Нейросеть nex-n2.5-pro синтезирует конспект с выбранными параметрами..."):
                    try:
                        api_client.synthesize_ai_study_guide(
                            current_doc["id"],
                            model="nex-agi/nex-n2.5-pro:free",
                            length_mode=length_mode_map[length_choice],
                            enrich_facts=enrich_facts,
                            creativity_mode=creativity_mode_map[creativity_choice],
                        )
                        st.success("Конспект успешно обновлен с учетом ваших настроек!")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Ошибка генерации через ИИ: {exc}")

        with col_algo_btn:
            if st.button("⚡ Быстрое алгоритмическое сведение", use_container_width=True):
                try:
                    api_client.export_document(current_doc["id"], export_format="MARKDOWN")
                    st.success("Алгоритмическая структура обновлена!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Ошибка генерации: {exc}")

        st.divider()

        # ----------------------------------------------------------------------
        # SUB-TABS: Note Text vs Flashcards vs Cloze Tests vs Quiz
        # ----------------------------------------------------------------------
        subtab_doc, subtab_cards, subtab_cloze, subtab_quiz = st.tabs([
            "📖 Конспект лекции",
            "🗂️ Карточки (Flashcards)",
            "🧩 Тренажер с пропусками",
            "🎯 Экспресс-квиз",
        ])

        # SUBTAB 1: LECTURE NOTE
        with subtab_doc:
            try:
                markdown_content = api_client.download_export(current_doc["id"], export_format="MARKDOWN")
            except Exception:
                markdown_content = f"# {current_doc['title']}\n\n*(Экспорт генерируется...)*\n"

            col_dl_md, col_dl_txt = st.columns(2)
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

            st.markdown("### 📖 Предварительный просмотр конспекта")
            st.markdown(markdown_content)

            with st.expander("🔍 Показать исходный код Markdown"):
                st.code(markdown_content, language="markdown")

        # Session state for interactive kit caching
        kit_cache_key = f"kit_{current_doc['id']}"
        if kit_cache_key not in st.session_state:
            st.session_state[kit_cache_key] = None

        # Helper to ensure kit is loaded
        def load_or_fetch_kit():
            if not st.session_state[kit_cache_key]:
                with st.spinner("🧠 Генерируем карточки и обучающие упражнения через ИИ..."):
                    try:
                        st.session_state[kit_cache_key] = api_client.get_interactive_kit(
                            current_doc["id"], model="nex-agi/nex-n2.5-pro:free"
                        )
                    except Exception as err:
                        st.error(f"Не удалось сгенерировать интерактивные материалы: {err}")

        # SUBTAB 2: FLASHCARDS
        with subtab_cards:
            st.markdown("### 🗂️ Карточки для запоминания терминов и понятий")
            st.caption("Интервальное повторение ключевых определений лекции.")

            col_gen_cards, col_refresh_cards = st.columns([7, 3])
            with col_refresh_cards:
                if st.button("🔄 Перегенерировать карточки", use_container_width=True, key="btn_regen_cards"):
                    st.session_state[kit_cache_key] = None
                    st.rerun()

            if st.session_state[kit_cache_key] is None:
                if st.button("✨ Создать карточки по этому конспекту", type="primary"):
                    load_or_fetch_kit()
                    st.rerun()
            else:
                kit_data = st.session_state[kit_cache_key]
                cards = kit_data.get("flashcards", [])
                if not cards:
                    st.info("Карточки не найдены. Нажмите «Перегенерировать карточки».")
                else:
                    card_idx_key = f"card_idx_{current_doc['id']}"
                    card_flipped_key = f"card_flip_{current_doc['id']}"
                    if card_idx_key not in st.session_state:
                        st.session_state[card_idx_key] = 0
                    if card_flipped_key not in st.session_state:
                        st.session_state[card_flipped_key] = False

                    current_idx = st.session_state[card_idx_key]
                    if current_idx >= len(cards):
                        current_idx = 0
                        st.session_state[card_idx_key] = 0

                    c = cards[current_idx]
                    total_c = len(cards)

                    st.progress((current_idx + 1) / total_c, text=f"Карточка {current_idx + 1} из {total_c} ({c.get('category', 'Термин')})")

                    # Flashcard Box Display
                    is_flipped = st.session_state[card_flipped_key]
                    card_bg = "#f8fafc" if not is_flipped else "#ecfdf5"
                    border_c = "#cbd5e1" if not is_flipped else "#10b981"

                    side_title = "❓ ВОПРОС / ПОНЯТИЕ" if not is_flipped else "💡 ОПРЕДЕЛЕНИЕ / ОТВЕТ"
                    main_text = c.get("front", "") if not is_flipped else c.get("back", "")
                    hint_text = f"💡 *Подсказка:* {c.get('hint')}" if (not is_flipped and c.get("hint")) else ""

                    card_html = f"""
                    <div style="background-color: {card_bg}; border: 2px solid {border_c}; border-radius: 12px; padding: 32px 24px; min-height: 180px; text-align: center; margin: 15px 0;">
                        <span style="font-size: 0.85em; font-weight: 700; color: #64748b; letter-spacing: 1px;">{side_title}</span>
                        <h3 style="margin-top: 15px; margin-bottom: 10px; color: #1e293b; font-size: 1.4em;">{main_text}</h3>
                        <p style="color: #64748b; font-size: 0.9em;">{hint_text}</p>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

                    col_prev, col_flip, col_next = st.columns([3, 4, 3])
                    with col_prev:
                        if st.button("⬅️ Предыдущая", disabled=(current_idx == 0), use_container_width=True):
                            st.session_state[card_idx_key] -= 1
                            st.session_state[card_flipped_key] = False
                            st.rerun()

                    with col_flip:
                        flip_label = "🔄 Перевернуть ответ" if not is_flipped else "↩️ Показать вопрос"
                        if st.button(flip_label, type="primary", use_container_width=True):
                            st.session_state[card_flipped_key] = not is_flipped
                            st.rerun()

                    with col_next:
                        if st.button("Следующая ➡️", disabled=(current_idx == total_c - 1), use_container_width=True):
                            st.session_state[card_idx_key] += 1
                            st.session_state[card_flipped_key] = False
                            st.rerun()

        # SUBTAB 3: CLOZE TESTS (FILL IN THE BLANKS)
        with subtab_cloze:
            st.markdown("### 🧩 Тренажер активного вспоминания (Вставка пропущенных слов)")
            st.caption("Проверьте, насколько точно вы помните ключевые понятия лекции без подсказок.")

            if st.session_state[kit_cache_key] is None:
                if st.button("✨ Сгенерировать упражнения с пропусками", type="primary", key="btn_gen_cloze"):
                    load_or_fetch_kit()
                    st.rerun()
            else:
                kit_data = st.session_state[kit_cache_key]
                cloze_items = kit_data.get("cloze_tests", [])
                if not cloze_items:
                    st.info("Упражнения с пропусками еще не сгенерированы. Нажмите «Сгенерировать».")
                else:
                    st.markdown(f"Всего упражнений: **{len(cloze_items)}**")
                    for i, item in enumerate(cloze_items):
                        with st.container(border=True):
                            st.markdown(f"**Задание {i + 1}:**")
                            sentence = item.get("sentence_with_blank") or item.get("text_with_blank", "")
                            st.markdown(f"### {sentence}")
                            if item.get("hint"):
                                st.caption(f"💡 Подсказка: {item.get('hint')}")

                            target = item.get("target_word") or (item.get("target_words", [""])[0] if item.get("target_words") else "")
                            options = item.get("options", [])

                            ans_key = f"cloze_ans_{current_doc['id']}_{i}"

                            if options:
                                user_choice = st.radio(
                                    "Выберите правильный вариант:",
                                    options=["(не выбрано)"] + options,
                                    key=ans_key,
                                    horizontal=True,
                                )
                                if user_choice != "(не выбрано)":
                                    if user_choice.strip().lower() == target.strip().lower():
                                        st.success(f"✅ Абсолютно верно! Полное предложение: *{item.get('full_sentence', '')}*")
                                    else:
                                        st.error(f"❌ Неверно. Правильное слово: **{target}**")
                            else:
                                user_input = st.text_input("Введите пропущенное слово:", key=ans_key)
                                if user_input:
                                    if user_input.strip().lower() == target.strip().lower():
                                        st.success(f"✅ В точку! Полное предложение: *{item.get('full_sentence', '')}*")
                                    else:
                                        st.warning(f"Правильный ответ: **{target}**")

        # SUBTAB 4: QUICK QUIZ
        with subtab_quiz:
            st.markdown("### 🎯 Экспресс-квиз по материалам конспекта")
            st.caption("Быстрая проверка понимания взаимосвязей и сущностей темы.")

            if st.session_state[kit_cache_key] is None:
                if st.button("✨ Сгенерировать экспресс-тест", type="primary", key="btn_gen_quiz"):
                    load_or_fetch_kit()
                    st.rerun()
            else:
                kit_data = st.session_state[kit_cache_key]
                quiz_items = kit_data.get("quiz", [])
                if not quiz_items:
                    st.info("Вопросы для теста отсутствуют.")
                else:
                    for qi, q in enumerate(quiz_items):
                        with st.container(border=True):
                            st.markdown(f"**Вопрос {qi + 1}:** {q.get('question')}")
                            opts = q.get("options", [])
                            q_key = f"quiz_{current_doc['id']}_{qi}"
                            user_pick = st.radio("Ваш ответ:", opts, index=None, key=q_key)

                            if user_pick is not None:
                                c_idx = q.get("correct_index", 0)
                                if c_idx < len(opts) and user_pick == opts[c_idx]:
                                    st.success(f"🎉 Правильно! {q.get('explanation', '')}")
                                else:
                                    correct_opt = opts[c_idx] if c_idx < len(opts) else ""
                                    st.error(f"Не совсем так. Правильный ответ: **{correct_opt}**. {q.get('explanation', '')}")


