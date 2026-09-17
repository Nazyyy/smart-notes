import pytest
import numpy as np
import torch
from app.ml.transformer_engine import TransformerHTREngine

def test_tta_predict_single_line_shape_resilience():
    engine = TransformerHTREngine(device="cpu")
    # Tiny crop should return empty gracefully
    empty_crop = np.ones((4, 4, 3), dtype=np.uint8) * 255
    text, conf = engine.predict_single_line(empty_crop, enable_tta=True)
    assert text == ""
    assert conf == 0.0

    # Synthetic line crop with text-like strokes
    line_crop = np.ones((48, 300, 3), dtype=np.uint8) * 255
    line_crop[20:28, 40:260] = 0  # black horizontal stripe
    text, conf = engine.predict_single_line(line_crop, enable_tta=True)
    assert isinstance(text, str)
    assert 0.0 <= conf <= 1.0

def test_tta_predict_batch():
    engine = TransformerHTREngine(device="cpu")
    crops = [
        np.ones((4, 4, 3), dtype=np.uint8) * 255,
        np.ones((4, 4, 3), dtype=np.uint8) * 255,
    ]
    results = engine.predict_batch(crops)
    assert len(results) == 2
    for text, conf in results:
        assert isinstance(text, str)
        assert 0.0 <= conf <= 1.0
