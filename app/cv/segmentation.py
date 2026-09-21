# ### FILE: app/cv/segmentation.py
"""
Handwritten Text Line Segmentation via Horizontal Projection Profiles (HPP).
Extracts line bounding boxes and cuts individual text line crops.
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from app.core.exceptions import ImageProcessingException
from app.core.logging import get_logger
from app.cv.advanced_segmentation import (
    compute_line_seams,
    extract_seam_carved_crop,
    straighten_text_line,
)

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


def suppress_pencil_underlines(rectified_bgr: Optional[np.ndarray], binary: np.ndarray) -> np.ndarray:
    """
    Remove horizontal graphite pencil lines (subject 1-line, predicate 2-line underlines)
    so they do not bridge adjacent text lines during vertical projection profile analysis.
    """
    if rectified_bgr is None or len(rectified_bgr.shape) != 3:
        return binary
    try:
        hsv = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        # Graphite pencil: low color saturation (gray/silver) and medium/dark brightness
        pencil_mask = (s < 32) & (v < 185) & (binary > 0)
        pencil_uint8 = (pencil_mask.astype(np.uint8)) * 255
        # Only remove long unbroken underline strokes (>= 36px), preserving individual letters/numbers
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (36, 1))
        pencil_lines = cv2.morphologyEx(pencil_uint8, cv2.MORPH_OPEN, h_kernel)
        pencil_lines = cv2.dilate(pencil_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)), iterations=1)
        clean = cv2.subtract(binary, pencil_lines)
        if np.sum(clean > 0) >= 120:
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
    min_peak_distance: int = 0,
) -> List[Tuple[int, int]]:
    """
    Detect vertical start and end row indices for individual text lines using HPP.
    Combines peak-valley analysis for dense/lined handwriting with thresholding fallback.
    Prevents intra-line splitting (ascenders vs x-height) and merges thin sub-line fragments.
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
        prominence = max(3.0, dynamic_range * 0.04)
        peak_dist = min_peak_distance if min_peak_distance > 0 else max(18, int(height * 0.016))
        peaks, _ = find_peaks(hpp, distance=peak_dist, prominence=prominence)

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

            # Valleys between consecutive peaks (pick median of valley floor for centered cuts)
            for i in range(len(peaks) - 1):
                p1, p2 = int(peaks[i]), int(peaks[i + 1])
                segment = hpp[p1:p2]
                min_v = np.min(segment)
                min_indices = np.where(segment == min_v)[0]
                v_idx = p1 + int(np.median(min_indices))
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

            raw_intervals: List[Tuple[int, int]] = []
            for i in range(len(peaks)):
                y1 = valleys[i]
                y2 = valleys[i + 1]
                if (y2 - y1) >= min_line_height:
                    raw_intervals.append((y1, y2))

            if raw_intervals:
                # Merge tiny sub-line splits (fragments narrower than 35% of median line height)
                heights = [y2 - y1 for y1, y2 in raw_intervals]
                med_h = float(np.median(heights)) if heights else 40.0
                min_h_thresh = max(14, int(med_h * 0.35))

                merged_peaks: List[Tuple[int, int]] = []
                i = 0
                while i < len(raw_intervals):
                    y1, y2 = raw_intervals[i]
                    curr_h = y2 - y1
                    if curr_h < min_h_thresh and i + 1 < len(raw_intervals):
                        ny1, ny2 = raw_intervals[i + 1]
                        # Only merge if combined height will not exceed normal line height
                        if (ny2 - y1) <= int(med_h * 1.35):
                            merged_peaks.append((y1, ny2))
                            i += 2
                            continue
                    merged_peaks.append((y1, y2))
                    i += 1
                return merged_peaks

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
    Clusters active ink columns and filters out distant margin dust.
    """
    clean = suppress_grid_lines(binary_slice)
    clean = remove_vertical_ruling_artifacts(clean)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    vpp = np.sum(clean > 0, axis=0)
    solid_cols = np.where(vpp >= min_content_threshold)[0]
    if len(solid_cols) == 0:
        return 0, binary_slice.shape[1]

    # Cluster active columns by inter-word distance
    clusters = []
    c_start = int(solid_cols[0])
    c_prev = int(solid_cols[0])
    for c in solid_cols[1:]:
        if c - c_prev > 45:
            clusters.append((c_start, c_prev))
            c_start = int(c)
        c_prev = int(c)
    clusters.append((c_start, c_prev))

    # Prune tiny isolated dust clusters (< 8px width and < 25 total ink)
    valid_clusters = []
    for s, e in clusters:
        cw = e - s + 1
        c_ink = int(np.sum(clean[:, s:e + 1] > 0))
        if cw >= 8 and c_ink >= 25:
            valid_clusters.append((s, e, c_ink))

    if not valid_clusters:
        return int(solid_cols[0]), int(solid_cols[-1])

    # Filter out edge clusters from neighboring notebook pages (e.g. left margin snippets)
    total_valid_ink = sum(c[2] for c in valid_clusters)
    if len(valid_clusters) >= 2:
        c0_s, c0_e, c0_ink = valid_clusters[0]
        c1_s, c1_e, _ = valid_clusters[1]
        gap_left = c1_s - c0_e
        if c0_s <= 75 and gap_left >= 40 and (c0_ink / max(1, total_valid_ink)) < 0.15:
            valid_clusters = valid_clusters[1:]

    # Filter out desk / edge paper fragments on right margin
    if len(valid_clusters) >= 2:
        c_last_s, c_last_e, c_last_ink = valid_clusters[-1]
        c_prev_s, c_prev_e, _ = valid_clusters[-2]
        gap_right = c_last_s - c_prev_e
        if (binary_slice.shape[1] - c_last_e) <= 80 and gap_right >= 40 and (c_last_ink / max(1, total_valid_ink)) < 0.15:
            valid_clusters = valid_clusters[:-1]

    x1 = max(0, valid_clusters[0][0] - 16)
    x2 = min(binary_slice.shape[1], valid_clusters[-1][1] + 18)
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
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    vpp = np.sum(clean > 0, axis=0)
    cols = np.where(vpp >= 3)[0]
    if len(cols) == 0:
        return crop, 0

    # Cluster active columns
    clusters = []
    c_start = int(cols[0])
    c_prev = int(cols[0])
    for c in cols[1:]:
        if c - c_prev > 45:
            clusters.append((c_start, c_prev))
            c_start = int(c)
        c_prev = int(c)
    clusters.append((c_start, c_prev))

    valid_clusters = []
    for s, e in clusters:
        cw = e - s + 1
        c_ink = int(np.sum(clean[:, s:e + 1] > 0))
        if cw >= 8 and c_ink >= 25:
            valid_clusters.append((s, e))

    if not valid_clusters:
        x1, x2 = int(cols[0]), int(cols[-1])
    else:
        x1 = valid_clusters[0][0]
        x2 = valid_clusters[-1][1]

    pad_left = max(0, x1 - 16)
    pad_right = min(crop.shape[1], x2 + 18)
    return crop[:, pad_left:pad_right], pad_left


def segment_text_lines(
    rectified_bgr: np.ndarray,
    binary_image: np.ndarray,
    padding: int = 6,
) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray], np.ndarray]:
    """
    Full line segmentation pipeline:
    1. Computes Horizontal Projection Profile (HPP) with adaptive Gaussian smoothing.
    2. Identifies line vertical spans using peak-valley analysis and merges sub-line splits.
    3. Finds tight horizontal bounds for each line.
    4. Tightens line crops to true ink boundaries, preserving margin details and ascenders/descenders.

    Returns:
        bboxes: List of (x, y, w, h) bounding boxes sorted top-to-bottom.
        crops: List of cropped BGR image slices for each line.
        hpp: 1D smoothed projection profile array.
    """
    try:
        img_h, img_w = rectified_bgr.shape[:2]
        clean_image = suppress_grid_lines(binary_image)
        clean_image = suppress_pencil_underlines(rectified_bgr, clean_image)
        eff_sigma = max(4.5, min(7.5, img_h / 160.0))
        hpp = compute_horizontal_projection_profile(clean_image, sigma=eff_sigma)
        raw_intervals = segment_line_intervals(hpp, min_line_height=12, min_gap=4)

        bboxes: List[Tuple[int, int, int, int]] = []
        crops: List[np.ndarray] = []

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
                # Merge true descender fragments (e.g. tails of р, у, д, з)
                if gap <= 5 and nh <= 18 and nink < 350:
                    y2 = ny2
                    i += 1
                else:
                    break
            final_h = y2 - y1
            final_ink = int(np.sum(clean_image[y1:y2, :] > 0))
            if final_ink >= 80 and final_h >= 12:
                merged_intervals.append((y1, y2))
            i += 1

        # Multi-line recursive splitting: guarantee that merged double-lines never reach OCR
        if merged_intervals:
            line_heights = [y2 - y1 for y1, y2 in merged_intervals]
            med_line_h = float(np.median(line_heights)) if line_heights else 40.0
            split_intervals: List[Tuple[int, int]] = []
            for y1, y2 in merged_intervals:
                curr_h = y2 - y1
                if curr_h >= 1.35 * med_line_h and curr_h >= 55:
                    slice_bin = clean_image[y1:y2, :]
                    row_ink = np.sum(slice_bin > 0, axis=1).astype(np.float32)
                    smooth_ink = gaussian_filter1d(row_ink, sigma=2.0)
                    lo = int(curr_h * 0.28)
                    hi = int(curr_h * 0.72)
                    if hi > lo:
                        mid_segment = smooth_ink[lo:hi]
                        min_mid = float(np.min(mid_segment))
                        max_top = float(np.max(smooth_ink[:lo])) if lo > 0 else 1.0
                        max_bot = float(np.max(smooth_ink[hi:])) if hi < curr_h else 1.0
                        if min_mid < 0.75 * min(max_top, max_bot) or min_mid < 45.0:
                            v_rel = lo + int(np.argmin(mid_segment))
                            split_y = y1 + v_rel
                            if (split_y - y1) >= 15 and (y2 - split_y) >= 15:
                                split_intervals.append((y1, split_y))
                                split_intervals.append((split_y, y2))
                                continue
                split_intervals.append((y1, y2))
            merged_intervals = split_intervals

        # Precompute non-linear separating seams between consecutive lines
        seams = compute_line_seams(clean_image, merged_intervals)

        for line_idx, (y1, y2) in enumerate(merged_intervals):
            # Drop top table / desk / border artifact only if right at the outer edge with no text
            if y2 <= int(img_h * 0.035) and y1 <= 4:
                continue
            # Drop bottom table border/shadow artifact only if at the extreme bottom edge
            if y2 >= img_h - 6 and y1 >= img_h - 40:
                continue


            h_line = y2 - y1
            # Add adaptive vertical padding so ascenders and descenders aren't clipped
            pad = max(padding, int(h_line * 0.14))
            pad_y1 = max(0, y1 - pad)
            pad_y2 = min(img_h, y2 + pad)

            padded_clean_slice = clean_image[pad_y1:pad_y2, :]
            x1, x2 = calculate_line_horizontal_bounds(padded_clean_slice)

            pad_x1 = max(0, x1)
            pad_x2 = min(img_w, x2)

            w = pad_x2 - pad_x1
            h = pad_y2 - pad_y1

            if w <= 20 or h <= 10:
                continue

            top_seam = seams[line_idx - 1] if line_idx > 0 and len(seams) >= line_idx else None
            bottom_seam = seams[line_idx] if line_idx < len(seams) else None

            # Non-linear seam carved crop preserving true ascenders/descenders
            raw_crop = extract_seam_carved_crop(
                rectified_bgr, clean_image, top_seam, bottom_seam, pad_y1, pad_y2, pad_x1, pad_x2
            )
            tight_crop, dx = tighten_line_crop(raw_crop)
            # Straighten slightly tilted/wavy lines for optimal TrOCR input
            straight_crop = straighten_text_line(tight_crop)

            final_w = straight_crop.shape[1]
            final_h = straight_crop.shape[0]
            final_x = pad_x1 + dx

            # Filter out true noise specks (< 24px width or < 8px height)
            if final_w < 24 or final_h < 8:
                continue

            crop_clean = suppress_grid_lines(binary_image[pad_y1:pad_y2, final_x:final_x + final_w])
            if int(np.sum(crop_clean > 0)) < 35:
                continue

            from app.cv.enhancer import suppress_notebook_grid_and_ruled_lines, pad_line_crop
            ready_crop = suppress_notebook_grid_and_ruled_lines(straight_crop)
            ready_crop = pad_line_crop(ready_crop)

            bboxes.append((final_x, pad_y1, final_w, final_h))
            crops.append(ready_crop)

        # Ensure top-to-bottom sorting by y coordinate
        sorted_pairs = sorted(zip(bboxes, crops), key=lambda item: item[0][1])
        sorted_bboxes = [item[0] for item in sorted_pairs]
        sorted_crops = [item[1] for item in sorted_pairs]

        logger.info("Successfully segmented %d text lines.", len(sorted_bboxes))
        return sorted_bboxes, sorted_crops, hpp

    except Exception as exc:
        logger.error("Text line segmentation failed: %s", exc)
        raise ImageProcessingException(step="segment_text_lines", reason=str(exc)) from exc
