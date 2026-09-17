import pytest
import numpy as np
from app.cv.enhancer import (
    enhance_stroke_sharpness,
    enhance_contrast_adaptive,
    suppress_paper_bleed_through,
    generate_tta_variants,
)

def test_stroke_sharpness():
    img = np.ones((50, 100, 3), dtype=np.uint8) * 128
    # Draw a line
    img[20:25, :, :] = 0
    enhanced = enhance_stroke_sharpness(img, strength=1.5)
    assert enhanced.shape == img.shape
    assert enhanced.dtype == np.uint8

def test_contrast_adaptive():
    img = np.random.randint(50, 200, (50, 100, 3), dtype=np.uint8)
    enhanced = enhance_contrast_adaptive(img, clip_limit=2.0)
    assert enhanced.shape == img.shape
    assert enhanced.dtype == np.uint8

def test_bleed_through_suppression():
    img = np.ones((60, 120, 3), dtype=np.uint8) * 240
    cleaned = suppress_paper_bleed_through(img)
    assert cleaned.shape == img.shape

def test_generate_tta_variants():
    img = np.ones((40, 150, 3), dtype=np.uint8) * 200
    variants = generate_tta_variants(img)
    assert len(variants) == 3
    names = [v[0] for v in variants]
    assert "base" in names
    assert "clahe" in names
    assert "sharpened" in names
