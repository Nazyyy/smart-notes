# ### FILE: frontend/components/viewer.py
"""
Computer Vision Intermediate Stages Visualizer Component.
Displays step-by-step images: raw, deskewed, shadow-suppressed, binarized, HPP, and overlay.
"""

from pathlib import Path
from typing import Any, Dict
import streamlit as st
from PIL import Image


def render_cv_stages_viewer(page: Dict[str, Any]) -> None:
    """Render interactive tabs showcasing each stage of OpenCV preprocessing."""
    st.subheader("🔍 Визуализация конвейера компьютерного зрения")

    debug_dir_str = page.get("debug_dir_path")
    if not debug_dir_str:
        st.info("Артефакты обработки для данной страницы еще не сгенерированы.")
        return

    debug_dir = Path(debug_dir_str)
    if not debug_dir.exists():
        st.warning(f"Каталог отладочных артефактов не найден: {debug_dir}")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Исходник",
        "2. Выравнивание (Deskew)",
        "3. Удаление теней",
        "4. Бинаризация",
        "5. Проекционный профиль (HPP)",
        "6. Разметка строк (Overlay)",
    ])

    with tab1:
        raw_path = Path(page.get("raw_image_path", ""))
        if raw_path.exists():
            st.image(Image.open(raw_path), caption="Оригинальная фотография конспекта", use_column_width=True)
        else:
            st.info("Исходный файл недоступен.")

    with tab2:
        deskewed_path = debug_dir / "02_rectified.jpg"
        angle = page.get("skew_angle", 0.0)
        st.caption(f"Вычисленный угол компенсации наклона: **{angle:.2f}°**")
        if deskewed_path.exists():
            st.image(Image.open(deskewed_path), caption="Геометрически выровненный документ", use_column_width=True)
        else:
            st.info("Этап выравнивания недоступен.")

    with tab3:
        shadow_path = debug_dir / "03_shadow_suppressed.jpg"
        st.caption("Морфологическая оценка фона и билатеральное подавление теней освещения.")
        if shadow_path.exists():
            st.image(Image.open(shadow_path), caption="Выровненное освещение и контраст", use_column_width=True)
        else:
            st.info("Этап подавления теней недоступен.")

    with tab4:
        bin_path = debug_dir / "04_binarized.png"
        st.caption("Адаптивная бинаризация чернил (комбинация Otsu + Gaussian Adaptive Thresholding).")
        if bin_path.exists():
            st.image(Image.open(bin_path), caption="Бинарная маска рукописного текста", use_column_width=True)
        else:
            st.info("Этап бинаризации недоступен.")

    with tab5:
        hpp_path = debug_dir / "05_projection_profile.png"
        st.caption("Горизонтальный проекционный профиль (HPP): пики соответствуют строкам, впадины — межстрочным интервалам.")
        if hpp_path.exists():
            st.image(Image.open(hpp_path), caption="Волновой график проекционного профиля", use_column_width=True)
        else:
            st.info("График HPP недоступен.")

    with tab6:
        overlay_path = debug_dir / "06_segmented_overlay.jpg"
        st.caption("Финальная сегментация: ограничивающие прямоугольники строк с порядковыми номерами и транскрипцией.")
        if overlay_path.exists():
            st.image(Image.open(overlay_path), caption="Сегментированные строки на документе", use_column_width=True)
        else:
            st.info("Разметка строк недоступна.")
