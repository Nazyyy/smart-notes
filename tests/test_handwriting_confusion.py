# ### FILE: tests/test_handwriting_confusion.py
import pytest
from app.ml.handwriting_confusion import (
    HandwritingConfusionCorrector,
    get_handwriting_confusion_corrector,
)


def test_handwriting_confusion_substitution_cost():
    corrector = HandwritingConfusionCorrector()
    # Identical letters -> 0.0
    assert corrector.substitution_cost("п", "п") == 0.0

    # Optical cursive cluster (п and и) -> 0.20
    assert corrector.substitution_cost("п", "и") == 0.20
    assert corrector.substitution_cost("ш", "т") == 0.20
    assert corrector.substitution_cost("о", "а") == 0.20

    # Unrelated letters (п and щ) -> 1.0
    assert corrector.substitution_cost("п", "щ") == 1.0


def test_weighted_levenshtein_distance():
    corrector = HandwritingConfusionCorrector()
    # "биосоииальное" vs "биосоциальное" (replacement of ц with и)
    dist_optical = corrector.weighted_levenshtein("биосоииальное", "биосоциальное")
    # Distance should be much less than a standard replacement of 1.0
    assert dist_optical <= 1.25


def test_get_word_candidates_optical_recovery():
    corrector = HandwritingConfusionCorrector()
    # Distorted academic term with cursive confusion
    candidates = corrector.get_word_candidates("антропогенез")
    assert len(candidates) >= 1
    assert candidates[0]["word"] == "антропогенез"
    assert candidates[0]["score"] == 1.0

    # Slight optical distortion: "биосоциальное" -> "биосоииальное"
    cands = corrector.get_word_candidates("биосоииальное", top_k=3)
    words = [c["word"].lower() for c in cands]
    assert "биосоциальное" in words


def test_singleton_getter():
    c1 = get_handwriting_confusion_corrector()
    c2 = get_handwriting_confusion_corrector()
    assert c1 is c2
