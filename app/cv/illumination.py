# ### FILE: app/cv/illumination.py
"""
Illumination Equalization, Shadow Removal, and Adaptive Binarization.
Restores non-uniform lighting and suppresses background paper noise.
"""

from typing import Tuple
import numpy as np
import cv2
from app.core.exceptions import ImageProcessingException
from app.core.logging import get_logger

logger = get_logger(__name__)


def remove_non_uniform_lighting(gray: np.ndarray, kernel_size: int = 51) -> np.ndarray:
    """
    Estimate background illumination using morphological closing with a large elliptical kernel,
    then apply division normalization to balance gradients across the page.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Closing captures background paper intensity beneath dark handwriting
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Prevent division by zero
    background = np.maximum(background, 1)

    # Division normalization: (gray / background) * 255
    normalized = cv2.divide(gray, background, scale=255)
    return normalized.astype(np.uint8)


def enhance_contrast_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to boost faint pencil/pen ink marks.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def suppress_shadows_and_denoise(image: np.ndarray) -> np.ndarray:
    """
    Suppresses uneven lighting and phone camera shadows:
    1. Converts to LAB color space.
    2. Applies CLAHE and morphological normalization on L-channel.
    3. Bilateral filtering to smooth gradient shadows while keeping sharp edges.
    """
    try:
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
        else:
            l_channel = image.copy()
            a_channel, b_channel = None, None

        # Normalize illumination on luminance
        norm_l = remove_non_uniform_lighting(l_channel, kernel_size=45)
        enhanced_l = enhance_contrast_clahe(norm_l, clip_limit=2.5)

        # Bilateral filter for edge-preserving denoising
        denoised_l = cv2.bilateralFilter(enhanced_l, d=7, sigmaColor=50, sigmaSpace=50)

        if a_channel is not None and b_channel is not None:
            merged_lab = cv2.merge([denoised_l, a_channel, b_channel])
            equalized_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
            return equalized_bgr

        return denoised_l

    except Exception as exc:
        logger.error("Shadow suppression failed: %s", exc)
        raise ImageProcessingException(step="suppress_shadows_and_denoise", reason=str(exc)) from exc


def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    """
    Produce high-contrast binary mask where ink text is 255 (foreground)
    and paper background is 0 (background).
    """
    try:
        # Morphological illumination correction
        normalized = remove_non_uniform_lighting(gray, kernel_size=41)

        # Adaptive Gaussian thresholding
        binary_adapt = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=31,
            C=11,
        )

        # Otsu thresholding as baseline
        _, binary_otsu = cv2.threshold(
            normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Bitwise combination: captures fine strokes while rejecting large dark blobs
        combined = cv2.bitwise_and(binary_adapt, binary_otsu)

        # Clean noise: remove isolated 1-pixel specks
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, clean_kernel)

        return cleaned

    except Exception as exc:
        logger.error("Adaptive binarization failed: %s", exc)
        raise ImageProcessingException(step="adaptive_binarize", reason=str(exc)) from exc
