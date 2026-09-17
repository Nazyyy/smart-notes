# ### FILE: tests/test_formula_recognizer.py
import pytest

from app.ml.formula_recognizer import AcademicFormulaRecognizer, get_formula_recognizer
from app.services.context_intelligence import ContextIntelligenceEngine


def test_formula_detection():
    rec = get_formula_recognizer()

    assert rec.is_formula_line("F = m * a") is True
    assert rec.is_formula_line("y = x^2 + 2x - 5") is True
    assert rec.is_formula_line("HCl + NaOH -> NaCl + H2O") is True
    assert rec.is_formula_line("E_k = m * v^2 / 2") is True
    assert rec.is_formula_line("Это обычный текст лекции по истории России") is False


def test_convert_math_to_latex():
    rec = get_formula_recognizer()

    # Fractions
    res = rec.convert_math_to_latex("(a + b) / (c - d)")
    assert "\\frac{a + b}{c - d}" in res

    # Roots and powers
    res = rec.convert_math_to_latex("sqrt(x + 1)")
    assert "\\sqrt{x + 1}" in res

    res = rec.convert_math_to_latex("x^2 + y^2 = R^2")
    assert "x^{2}" in res
    assert "y^{2}" in res

    # Greek letters and operators
    res = rec.convert_math_to_latex("Delta t != 0")
    assert "\\Delta" in res
    assert "\\neq" in res

    # Multiplication
    res = rec.convert_math_to_latex("F = m * a")
    assert "\\cdot" in res


def test_chemistry_conversion():
    rec = get_formula_recognizer()

    res = rec.convert_chemistry("H2SO4")
    assert "\\text{H}_2\\text{SO}_4" in res

    res = rec.convert_chemistry("2H2 + O2 -> 2H2O")
    assert "\\text{H}_2" in res
    assert "\\text{O}_2" in res


def test_format_line_block_and_inline():
    rec = get_formula_recognizer()

    # Block math
    block = rec.format_line("y = x^3 - 3x^2 + 2")
    assert block.startswith("$$")
    assert block.endswith("$$")
    assert "x^{3}" in block

    # Chemical reaction
    chem_block = rec.format_line("HCl + NaOH -> NaCl + H2O")
    assert chem_block.startswith("$$")
    assert "\\rightarrow" in chem_block

    # Text with inline chemical compound
    inline = rec.format_line("В пробирку добавили H2SO4 и нагрели.")
    assert "\\text{H}_2\\text{SO}_4" in inline
    assert "$" in inline


def test_context_intelligence_with_formulas():
    lines = [
        "Тема: Дифференциальные уравнения",
        "Рассмотрим уравнение движения:",
        "F = m * a",
        "Реакция синтеза:",
        "N2 + 3H2 -> 2NH3",
    ]
    md = ContextIntelligenceEngine.structure_into_markdown(lines, document_title="Физика")
    assert "$$" in md
    assert "\\cdot" in md or "m" in md
