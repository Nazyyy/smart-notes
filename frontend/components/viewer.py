# ### FILE: frontend/components/viewer.py
"""
Computer Vision Intermediate Stages Visualizer Component.
Displays step-by-step images: raw, deskewed, shadow-suppressed, binarized, HPP, and overlay.
"""

from pathlib import Path
from typing import Any, Dict
import streamlit as st
from PIL import Image


def render_photo_transformations(page: Dict[str, Any]) -> None:
    """Render original photo and all intermediate OpenCV cleanup stages."""
    st.subheader("📸 1. Исходная фотография и предварительные преобразования")

    raw_path_str = page.get("raw_image_path", "")
    raw_path = Path(raw_path_str) if raw_path_str else None

    debug_dir_str = page.get("debug_dir_path")
    debug_dir = Path(debug_dir_str) if debug_dir_str else None

    if not debug_dir or not debug_dir.exists():
        if raw_path and raw_path.exists():
            st.image(Image.open(raw_path), caption="Оригинальная фотография конспекта", use_column_width=True)
        else:
            st.info("Файлы изображения для данной страницы еще не сгенерированы.")
        return

    tab_orig, tab_deskew, tab_shadow, tab_bin = st.tabs([
        "📷 Оригинал",
        "📐 Выравнивание (Deskew)",
        "💡 Подавление теней",
        "🖤 Бинаризация (Савола)",
    ])

    with tab_orig:
        if raw_path and raw_path.exists():
            st.image(Image.open(raw_path), caption="Оригинальная фотография со смартфона", use_column_width=True)
        else:
            st.info("Исходный файл недоступен.")

    with tab_deskew:
        deskewed_path = debug_dir / "02_rectified.jpg"
        angle = page.get("skew_angle", 0.0)
        st.markdown(f"**Компенсация наклона:** вычислен угол поворота листа: `{angle:.2f}°`.")
        if deskewed_path.exists():
            st.image(Image.open(deskewed_path), caption="Геометрически выровненный конспект", use_column_width=True)
        else:
            st.info("Этап выравнивания недоступен.")

    with tab_shadow:
        shadow_path = debug_dir / "03_shadow_suppressed.jpg"
        st.markdown("**Нормализация освещения:** фильтрация перепадов вспышки телефона и удаление градиентных теней от руки.")
        if shadow_path.exists():
            st.image(Image.open(shadow_path), caption="Выровненный контраст и удаление теней", use_column_width=True)
        else:
            st.info("Этап подавления теней недоступен.")

    with tab_bin:
        bin_path = debug_dir / "04_binarized.png"
        st.markdown("**Адаптивная бинаризация:** выделение чернил ручки алгоритмом Саволы (k=0.22, r=128) с подавлением тетрадной клетки.")
        if bin_path.exists():
            st.image(Image.open(bin_path), caption="Бинарная маска рукописного текста", use_column_width=True)
        else:
            st.info("Этап бинаризации недоступен.")


def render_line_segmentation_stage(page: Dict[str, Any]) -> None:
    """Render line detection overlay, projection profile, and pipeline logs."""
    st.subheader("📐 2. Разметка и детекция строк текста")

    debug_dir_str = page.get("debug_dir_path")
    if not debug_dir_str:
        st.info("Данные сегментации для страницы еще не сформированы.")
        return

    debug_dir = Path(debug_dir_str)
    if not debug_dir.exists():
        st.warning(f"Каталог отладки не найден: {debug_dir}")
        return

    overlay_path = debug_dir / "06_segmented_overlay.jpg"
    hpp_path = debug_dir / "05_projection_profile.png"
    log_path = debug_dir / "00_pipeline.log"

    col_map, col_details = st.columns([7, 5])

    with col_map:
        st.markdown("#### 🗺️ Карта расположения строк (Overlay)")
        if overlay_path.exists():
            st.image(
                Image.open(overlay_path),
                caption="Каждая обнаруженная строка пронумерована. Номера соответствуют строкам во вкладке «3. Текст».",
                use_column_width=True,
            )
        else:
            st.info("Карта разметки строк недоступна.")

    with col_details:
        st.markdown("#### 📊 Волновой профиль проекций (HPP)")
        st.caption("Пики графика соответствуют центрам строк, впадины — межстрочным интервалам (Seam Carving).")
        if hpp_path.exists():
            st.image(Image.open(hpp_path), caption="Горизонтальный проекционный профиль", use_column_width=True)
        else:
            st.caption("График HPP отсутствует.")

        st.markdown("#### 📋 Журнал конвейера")
        if log_path.exists():
            with st.expander("Посмотреть полный журнал (CV & OCR Log)", expanded=True):
                st.code(log_path.read_text(encoding="utf-8"), language="text")
        else:
            st.caption("Журнал конвейера отсутствует.")


def render_cv_stages_viewer(page: Dict[str, Any]) -> None:
    """Backward-compatible fallback."""
    render_photo_transformations(page)
