# ### FILE: tests/test_contextual_and_antifragmentation.py
import pytest
import numpy as np
import cv2
from app.ml.transformer_engine import TransformerHTREngine
from app.services.context_intelligence import ContextIntelligenceEngine


def test_anti_fragmentation_normal_sentence():
    """Verify that normal inter-word gaps do not fragment a single line into pieces."""
    engine = TransformerHTREngine()

    # Create a synthetic text line with a 45px gap between words
    # (Width 400, Height 40)
    canvas = np.full((40, 400, 3), 255, dtype=np.uint8)
    # Word 1: x in [20, 150]
    canvas[15:25, 20:150] = 0
    # 45px white space gap: x in [150, 195]
    # Word 2: x in [195, 360]
    canvas[15:25, 195:360] = 0

    spans = engine._segment_line_into_spans(canvas)
    # Should remain as a single undivided line (not split into long and short fragments)
    assert len(spans) == 1
    assert spans[0][1] is False


def test_multi_column_table_detection():
    """Verify that a genuine wide column separator (e.g. 100px) on wide page splits as table."""
    engine = TransformerHTREngine()

    # Wide two-column notebook page (Width 800, Height 40)
    canvas = np.full((40, 800, 3), 255, dtype=np.uint8)
    # Column 1: x in [30, 300]
    canvas[12:28, 30:300] = 0
    # Wide 120px gap: x in [300, 420]
    # Column 2: x in [420, 750]
    canvas[12:28, 420:750] = 0

    spans = engine._segment_line_into_spans(canvas)
    assert len(spans) == 2
    assert spans[0][1] is True
    assert spans[1][1] is True


def test_cross_line_hyphenation_dewrapping():
    """Verify cross-line hyphenation stitching."""
    raw_lines = [
        "Антропогенез - теория возникно-",
        "вения человека и общества.",
    ]
    stitched = ContextIntelligenceEngine.dewrap_and_stitch_lines(raw_lines)
    assert len(stitched) == 1
    assert "возникновения" in stitched[0]
