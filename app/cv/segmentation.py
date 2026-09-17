# ### FILE: app/cv/segmentation.py
"""
Handwritten Text Line Segmentation via Horizontal Projection Profiles (HPP).
Extracts line bounding boxes and cuts individual text line crops.
"""

from typing import List, Tuple
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from app.core.exceptions import ImageProcessingException
from app.core.logging import get_logger

logger = get_logger(__name__)


def suppress_grid_lines(binary: np.ndarray) -> np.ndarray:
    """
    Remove thin checkered notebook grid lines (1-2px) before HPP analysis.
    Preserves thicker handwritten strokes while eliminating background grid peaks.
    """
    try:
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        grid = cv2.bitwise_or(h_lines, v_lines)
        clean = cv2.subtract(binary, grid)
        # Ensure we don't accidentally wipe out an image with faint pencil strokes
        if np.sum(clean > 0) >= 100:
            return clean
    except Exception:
        pass
    return binary


def compute_horizontal_projection_profile(binary: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """
    Compute horizontally summed projection profile smoothed with a 1D Gaussian filter.
    Input: binary image where foreground text pixels are > 0.
    Output: smoothed 1D array of length image_height.
    """
    clean = suppress_grid_lines(binary)
    # Count foreground pixels per row
    row_counts = np.sum(clean > 0, axis=1).astype(np.float32)
    # Smooth signal to bridge intra-character descenders and ascenders
    smoothed = gaussian_filter1d(row_counts, sigma=sigma)
    return smoothed



def segment_line_intervals(
    hpp: np.ndarray,
    min_line_height: int = 10,
    min_gap: int = 4,
) -> List[Tuple[int, int]]:
    """
    Detect vertical start and end row indices for individual text lines using HPP.
    Combines peak-valley analysis for dense/lined handwriting with thresholding fallback.
    """
    height = len(hpp)
    if height == 0:
        return []

    # Calculate dynamic range of HPP
    min_val = float(np.min(hpp))
    max_val = float(np.max(hpp))
    dynamic_range = max_val - min_val

    # Attempt peak detection if there is sufficient dynamic range
    if dynamic_range > 8.0:
        prominence = max(4.0, dynamic_range * 0.06)
        peaks, _ = find_peaks(hpp, distance=15, prominence=prominence)

        if len(peaks) >= 2:
            valleys: List[int] = []
            # Find upper boundary of first line
            p0 = int(peaks[0])
            top_bound = 0
            min_top = float(np.min(hpp[:p0 + 1]))
            thresh_top = min_top + (float(hpp[p0]) - min_top) * 0.15
            for y in range(p0, -1, -1):
                if hpp[y] <= thresh_top:
                    top_bound = y
                    break
            valleys.append(top_bound)

            # Valleys between consecutive peaks
            for i in range(len(peaks) - 1):
                p1, p2 = int(peaks[i]), int(peaks[i + 1])
                v_idx = p1 + int(np.argmin(hpp[p1:p2]))
                valleys.append(v_idx)

            # Bottom boundary of last line
            p_last = int(peaks[-1])
            bottom_bound = height
            min_bottom = float(np.min(hpp[p_last:]))
            thresh_bottom = min_bottom + (float(hpp[p_last]) - min_bottom) * 0.15
            for y in range(p_last, height):
                if hpp[y] <= thresh_bottom:
                    bottom_bound = y
                    break
            valleys.append(bottom_bound)

            intervals: List[Tuple[int, int]] = []
            for i in range(len(peaks)):
                y1 = valleys[i]
                y2 = valleys[i + 1]
                if (y2 - y1) >= min_line_height:
                    intervals.append((y1, y2))

            if intervals:
                return intervals

    # Fallback to adaptive valley thresholding for sparse / synthetic pages
    non_zero = hpp[hpp > 0]
    mean_val = float(np.mean(non_zero)) if len(non_zero) > 0 else 1.0
    threshold = max(2.0, mean_val * 0.08)

    in_line = False
    start_y = 0
    raw_intervals: List[Tuple[int, int]] = []

    for y in range(height):
        val = hpp[y]
        if not in_line and val >= threshold:
            in_line = True
            start_y = y
        elif in_line and val < threshold:
            in_line = False
            if (y - start_y) >= min_line_height:
                raw_intervals.append((start_y, y))

    if in_line and (height - start_y) >= min_line_height:
        raw_intervals.append((start_y, height))

    if not raw_intervals:
        return []

    # Merge intervals separated by tiny gaps (ascender/descender overlap)
    merged: List[Tuple[int, int]] = []
    current_start, current_end = raw_intervals[0]

    for nxt_start, nxt_end in raw_intervals[1:]:
        if nxt_start - current_end < min_gap:
            current_end = nxt_end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = nxt_start, nxt_end
    merged.append((current_start, current_end))

    return merged


def remove_vertical_ruling_artifacts(binary_slice: np.ndarray) -> np.ndarray:
    """
    Remove vertical line artifacts (e.g. margin red/blue ruling line, notebook grid rules)
    from a line slice without destroying genuine character strokes.
    Vertical ruling lines have very high aspect ratio (height >= 75% of slice, width <= 3px).
    """
    h, w = binary_slice.shape[:2]
    if h < 8 or w < 8:
        return binary_slice

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_slice, connectivity=8)
    clean = binary_slice.copy()
    for lbl in range(1, num_labels):
        comp_w = stats[lbl, cv2.CC_STAT_WIDTH]
        comp_h = stats[lbl, cv2.CC_STAT_HEIGHT]
        # Only eliminate if it is very tall and razor thin (grid or margin ruling line)
        if comp_w <= 3 and comp_h >= int(h * 0.70):
            clean[labels == lbl] = 0
    return clean


def calculate_line_horizontal_bounds(
    binary_slice: np.ndarray,
    min_content_threshold: int = 3,
) -> Tuple[int, int]:
    """
    Determine tight text bounds, ignoring checkered notebook grid lines and isolated margin specks.
    Preserves initial numbering, bullets, and trailing punctuation.
    """
    clean = suppress_grid_lines(binary_slice)
    clean = remove_vertical_ruling_artifacts(clean)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    vpp = np.sum(clean > 0, axis=0)
    solid_cols = np.where(vpp >= min_content_threshold)[0]
    if len(solid_cols) == 0:
        return 0, binary_slice.shape[1]

    x1 = int(solid_cols[0])
    x2 = int(solid_cols[-1])

    # Generous safety padding so outer strokes, digits, and punctuation are never truncated
    x1 = max(0, x1 - 12)
    x2 = min(binary_slice.shape[1], x2 + 14)
    return x1, x2


def tighten_line_crop(crop: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Refine crop boundaries by locating true ink clusters using Otsu binarization
    and pruning margin ruling artifacts while preserving initial characters/digits.
    Returns (tightened_crop, offset_x).
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 16:
        return crop, 0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    clean = suppress_grid_lines(otsu)
    clean = remove_vertical_ruling_artifacts(clean)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))

    vpp = np.sum(clean > 0, axis=0)
    cols = np.where(vpp >= 3)[0]
    if len(cols) == 0:
        return crop, 0

    x1, x2 = int(cols[0]), int(cols[-1])

    pad_left = max(0, x1 - 10)
    pad_right = min(crop.shape[1], x2 + 12)
    return crop[:, pad_left:pad_right], pad_left


def segment_text_lines(
    rectified_bgr: np.ndarray,
    binary_image: np.ndarray,
    padding: int = 4,
) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray], np.ndarray]:
    """
    Full line segmentation pipeline:
    1. Computes Horizontal Projection Profile (HPP).
    2. Identifies line vertical spans and merges descender micro-lines.
    3. Finds tight horizontal bounds for each line.
    4. Tightens line crops to true ink boundaries, preserving margin details.

    Returns:
        bboxes: List of (x, y, w, h) bounding boxes sorted top-to-bottom.
        crops: List of cropped BGR image slices for each line.
        hpp: 1D smoothed projection profile array.
    """
    try:
        img_h, img_w = rectified_bgr.shape[:2]
        hpp = compute_horizontal_projection_profile(binary_image, sigma=3.0)
        raw_intervals = segment_line_intervals(hpp, min_line_height=12, min_gap=4)

        bboxes: List[Tuple[int, int, int, int]] = []
        crops: List[np.ndarray] = []

        clean_image = suppress_grid_lines(binary_image)

        # Merge descender micro-lines and trailing letter tails into parent line
        merged_intervals: List[Tuple[int, int]] = []
        i = 0
        while i < len(raw_intervals):
            y1, y2 = raw_intervals[i]
            while i + 1 < len(raw_intervals):
                ny1, ny2 = raw_intervals[i + 1]
                nh = ny2 - ny1
                clean_next = clean_image[ny1:ny2, :]
                nink = int(np.sum(clean_next > 0))
                gap = ny1 - y2
                # Only merge true tiny descender fragments (e.g. tails of р, у, д, з)
                if gap <= 5 and nh <= 14 and nink < 300:
                    y2 = ny2
                    i += 1
                else:
                    break
            final_h = y2 - y1
            final_ink = int(np.sum(clean_image[y1:y2, :] > 0))
            if final_ink >= 100 and final_h >= 12:
                merged_intervals.append((y1, y2))
            i += 1

        for y1, y2 in merged_intervals:
            # Add vertical padding
            pad_y1 = max(0, y1 - padding)
            pad_y2 = min(img_h, y2 + padding)

            padded_clean_slice = clean_image[pad_y1:pad_y2, :]
            x1, x2 = calculate_line_horizontal_bounds(padded_clean_slice)

            pad_x1 = max(0, x1)
            pad_x2 = min(img_w, x2)

            w = pad_x2 - pad_x1
            h = pad_y2 - pad_y1

            if w <= 16 or h <= 8:
                continue

            raw_crop = rectified_bgr[pad_y1:pad_y2, pad_x1:pad_x2].copy()
            tight_crop, dx = tighten_line_crop(raw_crop)
            final_w = tight_crop.shape[1]
            final_x = pad_x1 + dx

            # Filter out isolated tiny debris/dust specks (< 40px width with negligible ink)
            if final_w < 50:
                gray_c = cv2.cvtColor(tight_crop, cv2.COLOR_BGR2GRAY) if len(tight_crop.shape) == 3 else tight_crop
                _, bin_c = cv2.threshold(gray_c, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                if int(np.sum(bin_c > 0)) < 80:
                    continue

            bboxes.append((final_x, pad_y1, final_w, h))
            crops.append(tight_crop)

        # Ensure top-to-bottom sorting by y coordinate
        sorted_pairs = sorted(zip(bboxes, crops), key=lambda item: item[0][1])
        sorted_bboxes = [item[0] for item in sorted_pairs]
        sorted_crops = [item[1] for item in sorted_pairs]

        logger.info("Successfully segmented %d text lines.", len(sorted_bboxes))
        return sorted_bboxes, sorted_crops, hpp

    except Exception as exc:
        logger.error("Text line segmentation failed: %s", exc)
        raise ImageProcessingException(step="segment_text_lines", reason=str(exc)) from exc
