# ### FILE: app/cv/enhancer.py
"""
Stroke Super-Resolution and Handwriting Image Enhancer.
Enhances handwritten strokes using unsharp masking, adaptive illumination normalization,
bleed-through suppression, and multi-view Test-Time Augmentation (TTA).
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np


def enhance_stroke_sharpness(image: np.ndarray, strength: float = 1.4) -> np.ndarray:
    """
    Crispen stroke contours and character edges using unsharp masking.
    Reveals subtle loops and letter junctions in handwritten Cyrillic text.
    """
    if image is None or image.size == 0:
        return image

    gaussian = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
    unsharp = cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)
    return np.clip(unsharp, 0, 255).astype(np.uint8)


def enhance_contrast_adaptive(image: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
    """
    Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE)
    tuned for faint ballpoint pen and pencil handwriting.
    """
    if image is None or image.size == 0:
        return image

    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l_chan)
        merged_lab = cv2.merge((cl, a_chan, b_chan))
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(image)


def suppress_paper_bleed_through(image: np.ndarray) -> np.ndarray:
    """
    Suppress ghost text and ink bleed-through from the back side of notebook paper.
    Uses morphological background estimation.
    """
    if image is None or image.size == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Structural element larger than typical stroke width to capture paper background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    # Background normalized difference
    diff = cv2.absdiff(bg, gray)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # Invert back to light paper background
    cleaned = 255 - norm

    if len(image.shape) == 3:
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    return cleaned


def suppress_notebook_grid_and_ruled_lines(image: np.ndarray) -> np.ndarray:
    """
    Non-destructive stroke enhancer and gentle background normalizer.
    Never alters dark handwritten strokes or bleaches blue/black ink.
    Only softly evens out paper background around the text.
    """
    if image is None or image.size == 0:
        return image

    # Gentle CLAHE contrast stabilization on luminance channel
    return enhance_contrast_adaptive(image, clip_limit=1.5)


def deskew_and_level_line_crop(crop: np.ndarray, max_angle: float = 12.0) -> np.ndarray:
    """
    Compute local baseline orientation of an individual handwritten text line crop
    and rotate it to a level horizontal baseline.
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 24:
        return crop

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    y_idx, x_idx = np.where(binary > 0)
    if len(x_idx) < 40:
        return crop

    pts_xy = np.column_stack([x_idx, y_idx]).astype(np.float32)
    rect = cv2.minAreaRect(pts_xy)
    box_w, box_h = rect[1]
    angle = rect[2]

    # OpenCV 4.5+ angle normalization
    if box_w < box_h:
        angle = angle - 90.0 if angle >= 45.0 else angle
    else:
        angle = angle if angle < 45.0 else angle - 90.0

    if abs(angle) < 0.6 or abs(angle) > max_angle:
        return crop

    h, w = crop.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    border_val = (255, 255, 255) if len(crop.shape) == 3 else 255
    rotated = cv2.warpAffine(
        crop, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val
    )
    return rotated


def pad_line_crop(crop: np.ndarray, pad_v: Optional[int] = None, pad_h: Optional[int] = None) -> np.ndarray:
    """
    Add clean white margin around cropped line to prevent descenders/ascenders
    from clipping against tensor bounds.
    """
    if crop is None or crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    pv = pad_v if pad_v is not None else max(6, int(h * 0.12))
    ph = pad_h if pad_h is not None else max(10, int(w * 0.04))

    border_val = (255, 255, 255) if len(crop.shape) == 3 else 255
    return cv2.copyMakeBorder(crop, pv, pv, ph, ph, cv2.BORDER_CONSTANT, value=border_val)


def generate_tta_variants(crop: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Generate diverse complementary visual representations of a text line crop
    for Test-Time Augmentation (TTA) beam search ensembling.
    
    Returns:
        List of (variant_name, image_array).
    """
    if crop is None or crop.size == 0:
        return []

    clean_crop = suppress_notebook_grid_and_ruled_lines(crop)
    level_crop = deskew_and_level_line_crop(clean_crop)

    variants: List[Tuple[str, np.ndarray]] = [
        ("base", level_crop),
        ("clahe", enhance_contrast_adaptive(level_crop, clip_limit=2.0)),
        ("sharpened", enhance_stroke_sharpness(level_crop, strength=1.2)),
    ]

    return variants

