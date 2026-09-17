# ### FILE: app/cv/advanced_segmentation.py
"""
Advanced Non-Linear Text Line Segmentation & Seam Carving Engine.
Resolves overlapping ascenders/descenders, detects curved baselines,
and extracts clean, stroke-preserved line crops for TrOCR recognition.
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2
from app.core.logging import get_logger

logger = get_logger(__name__)


def compute_energy_map(
    binary: np.ndarray,
    gray: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute 2D energy map where foreground ink and edges have high penalty,
    and white space has near-zero energy.
    """
    energy = np.zeros_like(binary, dtype=np.float32)
    # 1. High penalty for direct ink pixels
    energy[binary > 0] += 255.0

    # 2. Gradient magnitude penalty around stroke edges
    if gray is not None:
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        energy += cv2.normalize(grad_mag, None, 0.0, 100.0, cv2.NORM_MINMAX)
    else:
        # Morphological gradient of binary
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        energy[dilated > 0] += 50.0

    return energy


def find_optimal_horizontal_seam(
    energy: np.ndarray,
    y_min: int,
    y_max: int,
    center_weight: float = 0.05,
) -> np.ndarray:
    """
    Find optimal horizontal seam (path from x=0 to x=W-1) minimizing ink intersection
    between rows y_min and y_max using Dynamic Programming.
    Returns array of y-coordinates of shape (W,).
    """
    h, w = energy.shape[:2]
    y_min = max(0, int(y_min))
    y_max = min(h - 1, int(y_max))

    if y_max <= y_min:
        return np.full(w, y_min, dtype=np.int32)

    band_h = y_max - y_min + 1
    band_energy = energy[y_min : y_max + 1, :].copy()

    # Center bias penalty: encourage seam to stay near median valley height when in white space
    center_y = band_h / 2.0
    y_coords = np.arange(band_h, dtype=np.float32)
    center_penalty = center_weight * ((y_coords - center_y) ** 2)
    band_energy += center_penalty[:, np.newaxis]

    # Dynamic Programming cumulative energy matrix: M(y, x)
    dp = np.zeros((band_h, w), dtype=np.float32)
    dp[:, 0] = band_energy[:, 0]
    backtrack = np.zeros((band_h, w), dtype=np.int32)

    for x in range(1, w):
        for y in range(band_h):
            y_prev_min = max(0, y - 1)
            y_prev_max = min(band_h - 1, y + 1)
            candidates = dp[y_prev_min : y_prev_max + 1, x - 1]
            best_offset = int(np.argmin(candidates))
            best_prev_y = y_prev_min + best_offset

            dp[y, x] = band_energy[y, x] + candidates[best_offset]
            backtrack[y, x] = best_prev_y

    # Backtrack to reconstruct the optimal seam
    seam = np.zeros(w, dtype=np.int32)
    cur_y = int(np.argmin(dp[:, w - 1]))
    seam[w - 1] = cur_y + y_min

    for x in range(w - 2, -1, -1):
        cur_y = backtrack[cur_y, x + 1]
        seam[x] = cur_y + y_min

    return seam


def compute_line_seams(
    binary: np.ndarray,
    line_intervals: List[Tuple[int, int]],
    gray: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    Compute dividing seams between consecutive lines.
    Returns list of N-1 seams for N lines.
    """
    if len(line_intervals) <= 1:
        return []

    h, w = binary.shape[:2]
    energy = compute_energy_map(binary, gray)
    seams: List[np.ndarray] = []

    for i in range(len(line_intervals) - 1):
        _, cur_bottom = line_intervals[i]
        next_top, _ = line_intervals[i + 1]

        # Search band around the boundary
        mid = (cur_bottom + next_top) // 2
        band_size = max(10, abs(next_top - cur_bottom) + 12)
        y_min = max(0, mid - band_size // 2)
        y_max = min(h - 1, mid + band_size // 2)

        seam = find_optimal_horizontal_seam(energy, y_min, y_max)
        seams.append(seam)

    return seams


def extract_seam_carved_crop(
    image: np.ndarray,
    binary: np.ndarray,
    top_seam: Optional[np.ndarray],
    bottom_seam: Optional[np.ndarray],
    y1: int,
    y2: int,
    x1: int,
    x2: int,
) -> np.ndarray:
    """
    Extract line crop bounded by non-linear top and bottom seams.
    Masks out any foreign ink crossing the seam boundary to pure background.
    """
    h, w = image.shape[:2]
    x1 = max(0, x1)
    x2 = min(w, x2)
    y1 = max(0, y1)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return np.full((32, 100, 3), 255, dtype=np.uint8)

    crop = image[y1:y2, x1:x2].copy()
    crop_h, crop_w = crop.shape[:2]

    # Apply top seam mask if foreign descenders overlap
    if top_seam is not None:
        sub_top = top_seam[x1:x2] - y1
        for col in range(crop_w):
            cut_row = sub_top[col]
            if 0 < cut_row < crop_h:
                # Mask pixels above seam to white/background
                crop[:cut_row, col] = 255

    # Apply bottom seam mask if foreign ascenders overlap
    if bottom_seam is not None:
        sub_bottom = bottom_seam[x1:x2] - y1
        for col in range(crop_w):
            cut_row = sub_bottom[col]
            if 0 <= cut_row < crop_h:
                # Mask pixels below seam to white/background
                crop[cut_row:, col] = 255

    return crop


def straighten_text_line(crop: np.ndarray) -> np.ndarray:
    """
    Straighten slightly curved or tilted line using ink moment angle estimation.
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 20:
        return crop

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # Invert binary: text is 255
    _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(bin_inv > 0))
    if len(coords) < 50:
        return crop

    # Fit line through foreground stroke coordinates
    try:
        vx, vy, x0, y0 = cv2.fitLine(coords, cv2.DIST_L2, 0, 0.01, 0.01)
        angle_rad = np.arctan2(vx, vy)
        angle_deg = np.degrees(angle_rad) - 90.0

        # Only correct moderate slants within [-12°, +12°] to avoid flipping
        if -12.0 <= angle_deg <= 12.0 and abs(angle_deg) > 0.75:
            h, w = crop.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            straightened = cv2.warpAffine(
                crop,
                M,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255) if crop.ndim == 3 else 255,
            )
            return straightened
    except Exception as exc:
        logger.debug("Line straightening skipped: %s", exc)

    return crop
