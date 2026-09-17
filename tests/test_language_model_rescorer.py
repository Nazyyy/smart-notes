import pytest
from app.ml.language_model_rescorer import NGramLanguageModelRescorer, get_language_model_rescorer


def test_ngram_language_model_initialization():
    lm = NGramLanguageModelRescorer()
    assert lm.total_unigrams > 100
    assert lm.total_bigrams > 50
    assert "общества" in lm.vocab or "человека" in lm.vocab


def test_word_log_probability():
    lm = NGramLanguageModelRescorer()
    # "теория" after "антропогенез" should have high probability
    prob_context = lm.compute_word_log_prob("теория", prev_w1="антропогенез")
    prob_unseen = lm.compute_word_log_prob("абракадабра123", prev_w1="антропогенез")
    assert prob_context > prob_unseen


def test_gibberish_penalty():
    lm = NGramLanguageModelRescorer()
    # Valid Russian word vs random letters
    real_word = "ответственность"
    gibberish = "ыфвщзхъь"
    pen_real = lm.compute_char_perplexity_penalty(real_word)
    pen_gib = lm.compute_char_perplexity_penalty(gibberish)
    assert pen_real == 0.0
    assert pen_gib < 0.0  # Heavy negative penalty


def test_rescore_candidates():
    lm = NGramLanguageModelRescorer()
    candidates = [
        ("человек биосоциальное существо", -0.2),
        ("человек биосоииальное сушсство", -0.18),  # Slightly higher OCR prob, but gibberish
    ]
    best = lm.rescore_candidates(candidates)
    assert best == "человек биосоциальное существо"


def test_cross_line_context_scoring():
    lm = NGramLanguageModelRescorer()
    # Context: previous line ends with "теория"
    score_with_ctx = lm.score_sequence("возникновения человека", prev_context="Антропогенез - теория")
    score_without_ctx = lm.score_sequence("возникновения человека")
    assert score_with_ctx > score_without_ctx

    # Hyphenation: previous line ends with "обще-"
    score_hyphen_match = lm.score_sequence("ства", prev_context="развитие человеческого обще-")
    score_hyphen_wrong = lm.score_sequence("рыба", prev_context="развитие человеческого обще-")
    assert score_hyphen_match > score_hyphen_wrong


def test_singleton_getter():
    lm1 = get_language_model_rescorer()
    lm2 = get_language_model_rescorer()
    assert lm1 is lm2
