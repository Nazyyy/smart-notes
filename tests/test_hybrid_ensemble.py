# ### FILE: tests/test_hybrid_ensemble.py
import pytest
import numpy as np
from unittest.mock import MagicMock

from app.ml.hybrid_ensemble import ConsensusArbitrator, HybridEnsembleEngine


def test_consensus_repetitive_loop_detection():
    arbitrator = ConsensusArbitrator()
    
    # Repeating words
    assert arbitrator.has_repetitive_loop("слово слово слово слово") is True
    assert arbitrator.has_repetitive_loop("это обычное русское предложение без повторов") is False
    
    # Repeating characters
    assert arbitrator.has_repetitive_loop("аааааааааа") is True
    assert arbitrator.has_repetitive_loop("абвгабвгабвгабвг") is True


def test_consensus_arbitration_scenarios():
    arbitrator = ConsensusArbitrator()

    # Scenario 1: High confidence TrOCR
    text, conf, reason = arbitrator.arbitrate(
        trocr_text="Электрон движется по орбите",
        trocr_conf=0.95,
        crnn_text="Электрон движется по орбите",
        crnn_conf=0.70,
    )
    assert text == "Электрон движется по орбите"
    assert reason in ("trocr_high_confidence", "unanimous_agreement")

    # Scenario 2: TrOCR hallucination loop -> CRNN override
    text, conf, reason = arbitrator.arbitrate(
        trocr_text="орбите орбите орбите орбите орбите",
        trocr_conf=0.85,
        crnn_text="Уравнение Шредингера",
        crnn_conf=0.80,
    )
    assert text == "Уравнение Шредингера"
    assert reason == "crnn_override_repetitive_loop"

    # Scenario 3: Unanimous agreement boosts confidence
    text, conf, reason = arbitrator.arbitrate(
        trocr_text="Квантовая механика",
        trocr_conf=0.82,
        crnn_text="Квантовая механика",
        crnn_conf=0.84,
    )
    assert text == "Квантовая механика"
    assert conf > 0.84
    assert reason == "unanimous_agreement"

    # Scenario 4: Empty TrOCR
    text, conf, reason = arbitrator.arbitrate(
        trocr_text="",
        trocr_conf=0.0,
        crnn_text="Теория относительности",
        crnn_conf=0.75,
    )
    assert text == "Теория относительности"
    assert reason == "crnn_fallback_empty_trocr"


def test_hybrid_ensemble_engine_mock():
    mock_trocr = MagicMock()
    mock_crnn = MagicMock()

    # High confidence TrOCR should not call CRNN
    mock_trocr.predict_single_line.return_value = ("Закон сохранения энергии", 0.94)
    engine = HybridEnsembleEngine(transformer_engine=mock_trocr, crnn_engine=mock_crnn, consensus_threshold=0.88)

    dummy_crop = np.zeros((32, 128, 3), dtype=np.uint8)
    text, conf = engine.predict_single_line(dummy_crop)

    assert "сохранения" in text
    assert conf == 0.94
    mock_crnn.predict_batch.assert_not_called()

    # Low confidence TrOCR calls CRNN
    mock_trocr.predict_single_line.return_value = ("неразборчв", 0.40)
    mock_crnn.predict_batch.return_value = [("неразборчиво", 0.78)]

    text, conf = engine.predict_single_line(dummy_crop)
    assert mock_crnn.predict_batch.called
    assert conf > 0.40
