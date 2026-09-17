import pytest
from app.ml.vocabulary_binder import DomainVocabularyBinder

def test_vocabulary_binder_phrase_binding():
    binder = DomainVocabularyBinder
    # Test phrase mapping
    corrected = binder.clean_and_bind("Болюсное : внутривенно струйно")
    assert "Болюсное: внутривенно-струйно" in corrected

    corrected_diff = binder.clean_and_bind("1. Пассивная диффузия - поступление")
    assert "Пассивная диффузия - поступление в-в идет" in corrected_diff

    corrected_grad = binder.clean_and_bind("по градиенту концентрации")
    assert "по градиенту концентрации" in corrected_grad

def test_vocabulary_binder_word_replacements():
    binder = DomainVocabularyBinder
    res = binder.clean_and_bind("пинцитиз и вокзали")
    assert "пиноцитоз" in res
    assert "вакуоли" in res

def test_vocabulary_binder_hallucination_filtering():
    binder = DomainVocabularyBinder
    assert binder.clean_and_bind("X") == ""
    assert binder.clean_and_bind("??") == ""
    assert binder.clean_and_bind("З.") == ""
    assert binder.clean_and_bind("  .  ") == ""
    assert binder.clean_and_bind("М.З.") == ""

def test_vocabulary_binder_scientific_domain():
    binder = DomainVocabularyBinder
    res = binder.clean_and_bind("2. Фильтрация - процесс поступления вещества")
    assert "Фильтрация - процесс поступления вещества" in res
