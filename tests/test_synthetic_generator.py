# ### FILE: tests/test_synthetic_generator.py
import pytest
import numpy as np
from pathlib import Path

from app.ml.synthetic_generator import SyntheticHandwritingGenerator, ACADEMIC_CORPUS_TEMPLATES


def test_generator_init_and_fonts():
    gen = SyntheticHandwritingGenerator()
    assert len(gen.font_paths) > 0, "Should find font files in data/fonts"


@pytest.mark.parametrize("paper_type", ["blank", "grid", "ruled", "slant_ruled"])
def test_generate_line_paper_types(paper_type: str):
    gen = SyntheticHandwritingGenerator()
    target_h = 48
    img, text = gen.generate_line(
        text="Теорема Коши-Адамара",
        target_height=target_h,
        paper_type=paper_type,
    )

    assert isinstance(img, np.ndarray)
    assert img.shape[0] == target_h
    assert img.shape[1] >= 64
    assert img.shape[2] == 3
    assert text == "Теорема Коши-Адамара"
    assert img.dtype == np.uint8


def test_generate_math_and_chemistry():
    gen = SyntheticHandwritingGenerator()
    sample_text = "H2SO4 + 2NaOH = Na2SO4 + 2H2O"
    img, text = gen.generate_line(text=sample_text, target_height=52)

    assert img.shape[0] == 52
    assert img.shape[1] > 100
    assert text == sample_text


def test_corpus_coverage():
    assert len(ACADEMIC_CORPUS_TEMPLATES) >= 20
    for template in ACADEMIC_CORPUS_TEMPLATES:
        assert len(template.strip()) > 5
