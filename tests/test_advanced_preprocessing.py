# ### FILE: tests/test_advanced_preprocessing.py
"""
Unit tests for Advanced Preprocessing:
Sauvola Binarization, Notebook Grid Line Suppression, and Line Crop Deskewing.
"""

import numpy as np
import cv2
import pytest
from app.cv.illumination import sauvola_threshold, adaptive_binarize
from app.cv.enhancer import (
    suppress_notebook_grid_and_ruled_lines,
    deskew_and_level_line_crop,
    pad_line_crop,
)


def test_sauvola_threshold_under_shadows():
    # Create image with severe gradient shadow (left: bright 220, right: dark shadow 60)
    h, w = 120, 300
    x_coords = np.linspace(220, 60, w, dtype=np.float32)
    bg = np.tile(x_coords, (h, 1)).astype(np.uint8)

    # Draw dark handwritten text in both bright and dark halves
    cv2.putText(bg, "TEST1", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 20, 3)
    cv2.putText(bg, "TEST2", (180, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 20, 3)

    # Sauvola binarization
    bin_sauvola = sauvola_threshold(bg, window_size=31, k=0.20)

    # Verify that text pixels are detected (foreground = 255) in both halves
    # and background is 0 (not a huge black blob in the shadowed half)
    left_ink = np.sum(bin_sauvola[:, :150] == 255)
    right_ink = np.sum(bin_sauvola[:, 150:] == 255)

    assert left_ink > 100, "Left text should be segmented"
    assert right_ink > 100, "Right text under dark shadow must be segmented"
    # Ensure blank regions are not completely filled with black
    assert np.mean(bin_sauvola == 255) < 0.35, "Background should remain clean"


def test_suppress_notebook_grid_and_ruled_lines():
    # Create white canvas
    img = np.full((100, 200, 3), 255, dtype=np.uint8)

    # Draw cyan/blue notebook grid lines (H~90..110, BGR=(220, 200, 100))
    for y in range(20, 100, 20):
        cv2.line(img, (0, y), (200, y), (220, 200, 100), 1)
    for x in range(20, 200, 20):
        cv2.line(img, (x, 0), (x, 100), (220, 200, 100), 1)

    # Draw dark black handwritten stroke
    cv2.putText(img, "RUS", (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)

    cleaned = suppress_notebook_grid_and_ruled_lines(img)

    assert cleaned.shape == img.shape
    # Grid lines should be attenuated towards white (higher RGB sum than original grid)
    assert np.mean(cleaned) >= np.mean(img)


def test_deskew_and_level_line_crop():
    # Create line with tilted text (~10 degrees)
    img = np.full((60, 240), 255, dtype=np.uint8)
    cv2.putText(img, "LECTURE NOTE LINE", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 30, 2)

    # Artificially rotate by 8 degrees
    m = cv2.getRotationMatrix2D((120, 30), 8.0, 1.0)
    tilted = cv2.warpAffine(img, m, (240, 60), borderValue=255)

    level = deskew_and_level_line_crop(tilted)
    assert level.shape == tilted.shape


def test_pad_line_crop():
    crop = np.full((40, 150), 255, dtype=np.uint8)
    padded = pad_line_crop(crop, pad_v=8, pad_h=12)

    assert padded.shape[0] == 40 + 16
    assert padded.shape[1] == 150 + 24
