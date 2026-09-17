# ### FILE: app/cv/visualizer.py
"""
Computer Vision Visualizer and Debug Artifact Generator.
Renders intermediate stages, bounding box overlays, and HPP waveforms.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
from app.core.exceptions import ImageProcessingException
from app.core.logging import get_logger

logger = get_logger(__name__)


def render_projection_profile_image(
    hpp: np.ndarray,
    target_height: int,
    plot_width: int = 320,
) -> np.ndarray:
    """
    Render horizontal projection profile waveform into an image of dimensions (target_height, plot_width).
    The X-axis represents pixel count; Y-axis corresponds to document row Y.
    """
    canvas = np.full((target_height, plot_width, 3), 245, dtype=np.uint8)

    if len(hpp) == 0:
        return canvas

    # Resample or match height
    max_count = max(float(np.max(hpp)), 1.0)
    scale_x = (plot_width - 30) / max_count

    pts: List[Tuple[int, int]] = []
    step = max(1, len(hpp) // target_height)

    for y in range(0, min(len(hpp), target_height), step):
        val = hpp[y]
        x = int(val * scale_x) + 15
        pts.append((x, y))

    # Draw waveform
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], (220, 50, 50), 2, cv2.LINE_AA)

    # Draw axis guide line
    cv2.line(canvas, (15, 0), (15, target_height), (180, 180, 180), 1)
    cv2.putText(
        canvas, "HPP Waveform", (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA
    )

    return canvas


def draw_bounding_boxes_overlay(
    image: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    transcriptions: Optional[List[str]] = None,
    confidences: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Draw colored bounding boxes and index/transcription tags over the document.
    """
    overlay = image.copy()

    for idx, (x, y, w, h) in enumerate(bboxes):
        # Draw bounding rectangle in vibrant green
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 200, 70), 2)

        # Label tag
        label_text = f"#{idx}"
        if transcriptions and idx < len(transcriptions):
            preview = transcriptions[idx][:18]
            if len(transcriptions[idx]) > 18:
                preview += "..."
            conf_str = f" ({confidences[idx]:.2f})" if confidences and idx < len(confidences) else ""
            label_text = f"#{idx}: {preview}{conf_str}"

        # Background badge for text legibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

        badge_y1 = max(0, y - th - 6)
        badge_y2 = y
        badge_x1 = x
        badge_x2 = min(overlay.shape[1], x + tw + 8)

        cv2.rectangle(overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), (0, 200, 70), -1)
        cv2.putText(
            overlay,
            label_text,
            (x + 4, y - 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return overlay


def save_pipeline_debug_artifacts(
    debug_dir: Path,
    raw_image: np.ndarray,
    rectified_image: np.ndarray,
    shadow_suppressed: np.ndarray,
    binary_image: np.ndarray,
    hpp: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    crops: List[np.ndarray],
    transcriptions: Optional[List[str]] = None,
    confidences: Optional[List[float]] = None,
) -> None:
    """
    Save all intermediate debug images and line crops to the specified debug directory.
    """
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        crops_dir = debug_dir / "lines"
        crops_dir.mkdir(parents=True, exist_ok=True)

        # Save stage images
        cv2.imwrite(str(debug_dir / "01_raw.jpg"), raw_image)
        cv2.imwrite(str(debug_dir / "02_rectified.jpg"), rectified_image)
        cv2.imwrite(str(debug_dir / "03_shadow_suppressed.jpg"), shadow_suppressed)
        cv2.imwrite(str(debug_dir / "04_binarized.png"), binary_image)

        # Save HPP graph
        hpp_img = render_projection_profile_image(hpp, target_height=rectified_image.shape[0])
        cv2.imwrite(str(debug_dir / "05_projection_profile.png"), hpp_img)

        # Save annotated overlay
        annotated = draw_bounding_boxes_overlay(
            rectified_image, bboxes, transcriptions=transcriptions, confidences=confidences
        )
        cv2.imwrite(str(debug_dir / "06_segmented_overlay.jpg"), annotated)

        # Save individual line crops
        for idx, crop in enumerate(crops):
            crop_path = crops_dir / f"line_{idx:03d}.jpg"
            cv2.imwrite(str(crop_path), crop)

        logger.info("Saved pipeline debug artifacts in %s", debug_dir)

    except Exception as exc:
        logger.error("Failed saving debug artifacts: %s", exc)
        raise ImageProcessingException(step="save_pipeline_debug_artifacts", reason=str(exc)) from exc
