# ### FILE: tests/test_paragraph_corrector.py
import pytest

from app.services.paragraph_context_corrector import ParagraphContextCorrector, get_paragraph_context_corrector


def test_stitch_hyphenated_words():
    corrector = get_paragraph_context_corrector()
    lines = [
        "поступления ЛС в кровеносную и лимфати-",
        "ческую систему организма.",
        "Обычная строка без переноса.",
    ]
    res = corrector.stitch_hyphenated_words(lines)
    assert len(res) == 2
    assert "лимфатическую" in res[0]
    assert "Обычная строка" in res[1]


def test_bind_hanging_prepositions():
    corrector = get_paragraph_context_corrector()
    lines = [
        "Процесс проникновения молекул в",
        "клеточную мембрану.",
    ]
    res = corrector.bind_hanging_prepositions(lines)
    # 'в' should be bound with 'клеточную мембрану'
    assert any("в клеточную мембрану" in l for l in res)


def test_merge_broken_sentences():
    corrector = get_paragraph_context_corrector()
    lines = [
        "Человек представляет собой биосоциальное существо,",
        "высшую ступень развития живых организмов на Земле.",
        "2. Сознание и разум.",
    ]
    res = corrector.merge_broken_sentences(lines)
    assert len(res) == 2
    assert "биосоциальное существо, высшую ступень" in res[0]


def test_restore_capitalization():
    corrector = get_paragraph_context_corrector()
    text = "первое предложение. второе предложение! третье предложение? четвертое."
    fixed = corrector.restore_capitalization(text)
    assert fixed.startswith("Первое")
    assert "Второе" in fixed
    assert "Третье" in fixed
    assert "Четвертое" in fixed


def test_full_pipeline():
    corrector = get_paragraph_context_corrector()
    raw_lines = [
        "абсорбция - процесс поступления в-в из",
        "места введения в кровенос-",
        "ную систему.",
        "если ускорить - кладем грелку.",
    ]
    processed = corrector.process_document_lines(raw_lines)
    assert any("кровеносную" in l for l in processed)
    assert any("Если ускорить" in l for l in processed)
