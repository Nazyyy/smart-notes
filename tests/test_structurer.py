# ### FILE: tests/test_structurer.py
"""
Unit Tests for NoteStructurerService.
Verifies layout-driven Markdown formatting, LaTeX block synthesis, and multi-format exports.
"""

from uuid import uuid4
import pytest

from app.models.entities import TextLine, ExportFormat
from app.services.structurer_service import NoteStructurerService


def create_mock_line(text: str, line_index: int, bbox_x: int = 50, bbox_h: int = 24) -> TextLine:
    """Helper to instantiate TextLine for testing."""
    return TextLine(
        id=uuid4(),
        page_id=uuid4(),
        line_index=line_index,
        bbox_x=bbox_x,
        bbox_y=100 + line_index * 40,
        bbox_w=400,
        bbox_h=bbox_h,
        recognized_text=text,
        confidence=0.95,
    )


def test_classify_line_types():
    """Verify semantic classification heuristics."""
    structurer = NoteStructurerService()
    avg_h = 20.0

    # Major header by keyword
    t1, _ = structurer.classify_line("Лекция 5: Ряды Фурье", 400, 25, avg_h)
    assert t1 == "h2"

    # Header by size
    t2, _ = structurer.classify_line("КРАТКИЙ ЗАГОЛОВОК", 300, 32, avg_h)
    assert t2 == "h1"

    # Math formula
    t3, _ = structurer.classify_line("y = \\int_0^1 x^2 dx", 300, 20, avg_h)
    assert t3 == "math"

    # Bullet point
    t4, _ = structurer.classify_line("- Первое базовое свойство", 350, 20, avg_h)
    assert t4 == "bullet"

    # Regular paragraph
    t5, _ = structurer.classify_line("Рассмотрим произвольное линейное пространство.", 450, 20, avg_h)
    assert t5 == "paragraph"


def test_structure_lines_to_markdown():
    """Verify Markdown document compilation."""
    structurer = NoteStructurerService()

    lines = [
        create_mock_line("Лекция 1: Введение", 0, bbox_x=50, bbox_h=30),
        create_mock_line("Ниже приведены основные аксиомы:", 1, bbox_x=50, bbox_h=20),
        create_mock_line("- Аксиома коммутативности", 2, bbox_x=80, bbox_h=20),
        create_mock_line("- Аксиома ассоциативности", 3, bbox_x=80, bbox_h=20),
        create_mock_line("E = m * c^2", 4, bbox_x=50, bbox_h=22),
    ]

    markdown = structurer.structure_lines_to_markdown(lines, document_title="Физика")

    assert "# Физика" in markdown
    assert "### Лекция 1: Введение" in markdown or "## Лекция 1: Введение" in markdown
    assert "- Аксиома коммутативности" in markdown
    assert "$$\nE = m * c^2\n$$" in markdown


def test_export_formats():
    """Verify dispatch for MARKDOWN, TXT, LATEX, and JSON exports."""
    structurer = NoteStructurerService()
    lines = [
        create_mock_line("Заголовок документа", 0),
        create_mock_line("Текст параграфа конспекта.", 1),
    ]

    md_out = structurer.export(lines, ExportFormat.MARKDOWN, "Тест")
    assert "# Тест" in md_out

    txt_out = structurer.export(lines, ExportFormat.TXT, "Тест")
    assert "=== Тест ===" in txt_out
    assert "Текст параграфа конспекта." in txt_out

    latex_out = structurer.export(lines, ExportFormat.LATEX, "Тест")
    assert "\\begin{document}" in latex_out

    json_out = structurer.export(lines, ExportFormat.JSON, "Тест")
    assert '"title": "Тест"' in json_out
    assert '"lines":' in json_out
