# ### FILE: app/cv/enhancer.py
"""
Stroke Super-Resolution and Handwriting Image Enhancer.
Enhances handwritten strokes using unsharp masking, adaptive illumination normalization,
bleed-through suppression, and multi-view Test-Time Augmentation (TTA).
"""

from typing import List, Tuple
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


def generate_tta_variants(crop: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Generate diverse complementary visual representations of a text line crop
    for Test-Time Augmentation (TTA) beam search ensembling.
    
    Returns:
        List of (variant_name, image_array).
    """
    if crop is None or crop.size == 0:
        return []

    variants: List[Tuple[str, np.ndarray]] = [
        ("base", crop),
        ("clahe", enhance_contrast_adaptive(crop, clip_limit=2.0)),
        ("sharpened", enhance_stroke_sharpness(crop, strength=1.2)),
    ]

    return variants
