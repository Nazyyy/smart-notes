# ### FILE: app/cv/scanner.py
"""
Document Scanner and Optical Geometry Rectification.
Includes 4-point perspective warp and text deskewing.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
from app.core.exceptions import ImageProcessingException
from app.core.logging import get_logger

logger = get_logger(__name__)


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order quadrilateral coordinates in order:
    [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Perform perspective transform to obtain top-down overhead perspective.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    if max_width <= 0 or max_height <= 0:
        return image

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))
    return warped


def detect_document_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect largest quadrilateral contour representing the page boundary.
    Returns array of 4 points or None if no clear page boundary is found.
    """
    h, w = image.shape[:2]
    min_area = (h * w) * 0.15

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edged, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")
        elif 4 < len(approx) <= 8 and cv2.isContourConvex(approx):
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            return box.astype("float32")

    # Fallback to bright paper region detection if edge contours are broken by fingers or shadows
    try:
        _, paper_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_p = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        paper_mask = cv2.morphologyEx(paper_thresh, cv2.MORPH_CLOSE, kernel_p)
        paper_cnts, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if paper_cnts:
            best_paper = max(paper_cnts, key=cv2.contourArea)
            paper_area = cv2.contourArea(best_paper)
            total_img_area = float(h * w)
            # Only use if page is a distinct document occupying 25% to 95% of photo
            if (total_img_area * 0.25) <= paper_area <= (total_img_area * 0.96):
                rect = cv2.minAreaRect(best_paper)
                box = cv2.boxPoints(rect)
                return box.astype("float32")
    except Exception as exc:
        logger.debug("Paper region fallback failed: %s", exc)


    return None


def calculate_skew_angle(binary_image: np.ndarray) -> float:
    """
    Determine dominant skew angle of handwritten text lines in degrees.
    Uses cv2.minAreaRect on foreground pixel coordinates.
    """
    # Foreground pixels are 255
    coords = np.column_stack(np.where(binary_image > 0))
    if len(coords) < 100:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]

    # Adjust angle semantics
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle

    # Filter out extreme false angles (> 45 deg)
    if abs(angle) > 45.0:
        return 0.0

    return float(angle)


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image by angle (in degrees) about its center with white background padding.
    """
    if abs(angle) < 0.1:
        return image.copy()

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])

    # Compute new bounding dimensions
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    m[0, 2] += (new_w / 2) - center[0]
    m[1, 2] += (new_h / 2) - center[1]

    border_value = (255, 255, 255) if len(image.shape) == 3 else 255
    rotated = cv2.warpAffine(
        image,
        m,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return rotated


def rectify_document_geometry(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Full optical rectification pipeline:
    1. Detect page quad and warp perspective if detected.
    2. Estimate skew angle and rotate to horizontal baseline.
    Returns: (rectified_bgr_image, skew_angle_degrees)
    """
    try:
        quad = detect_document_contour(image)
        warped = four_point_transform(image, quad) if quad is not None else image.copy()

        # Generate quick binary mask for skew estimation
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped.copy()
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        skew_angle = calculate_skew_angle(binary)
        rectified = rotate_image(warped, skew_angle)

        return rectified, skew_angle

    except Exception as exc:
        logger.error("Optical geometry rectification failed: %s", exc)
        raise ImageProcessingException(step="rectify_document_geometry", reason=str(exc)) from exc
