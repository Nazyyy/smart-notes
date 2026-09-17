import numpy as np
import cv2
import pytest
from app.cv.advanced_segmentation import (
    compute_energy_map,
    find_optimal_horizontal_seam,
    compute_line_seams,
    extract_seam_carved_crop,
    straighten_text_line,
)


def test_energy_map_computation():
    binary = np.zeros((100, 200), dtype=np.uint8)
    binary[40:60, 20:180] = 255
    energy = compute_energy_map(binary)
    assert energy.shape == (100, 200)
    assert np.max(energy[45, :]) > 100.0


def test_find_optimal_horizontal_seam():
    energy = np.zeros((60, 100), dtype=np.float32)
    # Put a barrier of high energy in the middle
    energy[25:35, 30:70] = 500.0
    seam = find_optimal_horizontal_seam(energy, y_min=10, y_max=50)
    assert len(seam) == 100
    assert np.all(seam >= 10) and np.all(seam <= 50)
    # The seam should avoid the barrier at x=50
    assert seam[50] < 25 or seam[50] > 34


def test_extract_seam_carved_crop():
    image = np.full((120, 200, 3), 255, dtype=np.uint8)
    binary = np.zeros((120, 200), dtype=np.uint8)
    # Line 1
    image[20:40, 10:190] = 0
    binary[20:40, 10:190] = 255
    # Line 2
    image[70:90, 10:190] = 0
    binary[70:90, 10:190] = 255

    seams = compute_line_seams(binary, [(15, 45), (65, 95)])
    assert len(seams) == 1
    crop = extract_seam_carved_crop(image, binary, None, seams[0], 10, 50, 5, 195)
    assert crop.shape[0] == 40
    assert crop.shape[1] == 190


def test_straighten_text_line():
    crop = np.full((40, 200, 3), 255, dtype=np.uint8)
    # Draw a tilted line of text
    cv2.putText(crop, "Тестовая строка", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    straight = straighten_text_line(crop)
    assert straight.shape == crop.shape
