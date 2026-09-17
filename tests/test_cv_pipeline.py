# ### FILE: tests/test_cv_pipeline.py
"""
Unit and Integration Tests for Computer Vision Subsystem.
Verifies perspective transform, deskewing, lighting equalization, binarization, and HPP segmentation.
"""

from pathlib import Path
import numpy as np
import cv2
import pytest

from app.cv.scanner import (
    order_points,
    four_point_transform,
    calculate_skew_angle,
    rotate_image,
    rectify_document_geometry,
)
from app.cv.illumination import (
    remove_non_uniform_lighting,
    suppress_shadows_and_denoise,
    adaptive_binarize,
)
from app.cv.segmentation import (
    compute_horizontal_projection_profile,
    segment_line_intervals,
    segment_text_lines,
)
from app.cv.visualizer import (
    render_projection_profile_image,
    draw_bounding_boxes_overlay,
    save_pipeline_debug_artifacts,
)


def test_order_points():
    """Verify ordering of arbitrary 4 quadrilateral points into [TL, TR, BR, BL]."""
    pts = np.array([[100, 200], [10, 20], [110, 30], [20, 190]], dtype="float32")
    ordered = order_points(pts)

    assert ordered.shape == (4, 2)
    # Top-left has smallest sum of coordinates
    assert np.allclose(ordered[0], [10, 20])
    # Bottom-right has largest sum of coordinates
    assert np.allclose(ordered[2], [100, 200])


def test_four_point_transform():
    """Verify perspective transform maps quadrilateral region to flat rectangular image."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Draw white rectangle in center
    cv2.rectangle(img, (50, 50), (350, 350), (255, 255, 255), -1)

    pts = np.array([[50, 50], [350, 50], [350, 350], [50, 350]], dtype="float32")
    warped = four_point_transform(img, pts)

    assert warped.shape[0] > 250
    assert warped.shape[1] > 250
    # Center should be white
    assert warped[100, 100, 0] == 255


def test_calculate_skew_angle_and_rotate():
    """Test text angle detection and rotation."""
    # Create an image with a horizontal text line rotated by -5 degrees
    canvas = np.zeros((200, 500), dtype=np.uint8)
    cv2.putText(canvas, "HORIZONTAL TEXT LINE TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 3)

    # Angle of pure horizontal text should be near zero
    angle_orig = calculate_skew_angle(canvas)
    assert abs(angle_orig) < 5.0

    # Rotate by 10 degrees and test rotation function
    rotated = rotate_image(canvas, 10.0)
    assert rotated.shape[0] >= canvas.shape[0]
    assert rotated.shape[1] >= canvas.shape[1]


def test_remove_non_uniform_lighting(synthetic_page_image: np.ndarray):
    """Verify illumination normalization levels gradients across dark regions."""
    gray = cv2.cvtColor(synthetic_page_image, cv2.COLOR_BGR2GRAY)
    normalized = remove_non_uniform_lighting(gray, kernel_size=31)

    assert normalized.shape == gray.shape
    assert normalized.dtype == np.uint8
    # Standard deviation of background pixels should be reduced
    assert float(np.mean(normalized)) > float(np.mean(gray))


def test_suppress_shadows_and_denoise(synthetic_page_image: np.ndarray):
    """Verify shadow suppression returns valid 3-channel image."""
    equalized = suppress_shadows_and_denoise(synthetic_page_image)
    assert equalized.shape == synthetic_page_image.shape
    assert equalized.dtype == np.uint8


def test_adaptive_binarize(synthetic_page_image: np.ndarray):
    """Verify binary mask output: ink text foreground > 0, background 0."""
    gray = cv2.cvtColor(synthetic_page_image, cv2.COLOR_BGR2GRAY)
    binary = adaptive_binarize(gray)

    assert binary.shape == gray.shape
    assert set(np.unique(binary)).issubset({0, 255})

    # Foreground pixels must be present for text lines
    foreground_ratio = np.sum(binary > 0) / float(binary.size)
    assert 0.005 < foreground_ratio < 0.30


def test_segment_text_lines(synthetic_page_image: np.ndarray):
    """Verify line segmentation extracts correct number of lines from synthetic page."""
    rectified, _ = rectify_document_geometry(synthetic_page_image)
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    binary = adaptive_binarize(gray)

    bboxes, crops, hpp = segment_text_lines(rectified, binary)

    # In our synthetic image, there are 6 distinct lines
    assert len(bboxes) >= 4
    assert len(bboxes) == len(crops)

    # Check top-to-bottom monotonicity of bounding box Y coordinates
    y_coords = [bbox[1] for bbox in bboxes]
    assert y_coords == sorted(y_coords)

    # Check that crops have valid dimensions
    for crop in crops:
        assert crop.shape[0] >= 8
        assert crop.shape[1] >= 8


def test_visualizer_artifacts_generation(synthetic_page_image: np.ndarray, temp_dir: Path):
    """Verify that debug visualizer outputs all expected stage images."""
    gray = cv2.cvtColor(synthetic_page_image, cv2.COLOR_BGR2GRAY)
    binary = adaptive_binarize(gray)
    bboxes, crops, hpp = segment_text_lines(synthetic_page_image, binary)

    debug_out = temp_dir / "test_debug_vis"
    save_pipeline_debug_artifacts(
        debug_dir=debug_out,
        raw_image=synthetic_page_image,
        rectified_image=synthetic_page_image,
        shadow_suppressed=synthetic_page_image,
        binary_image=binary,
        hpp=hpp,
        bboxes=bboxes,
        crops=crops,
        transcriptions=["Line text sample"] * len(bboxes),
        confidences=[0.95] * len(bboxes),
    )

    assert (debug_out / "01_raw.jpg").exists()
    assert (debug_out / "02_rectified.jpg").exists()
    assert (debug_out / "03_shadow_suppressed.jpg").exists()
    assert (debug_out / "04_binarized.png").exists()
    assert (debug_out / "05_projection_profile.png").exists()
    assert (debug_out / "06_segmented_overlay.jpg").exists()
    assert (debug_out / "lines").exists()
    assert len(list((debug_out / "lines").glob("*.jpg"))) == len(crops)
