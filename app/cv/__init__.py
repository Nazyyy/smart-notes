# ### FILE: app/cv/__init__.py
"""
Computer Vision Subsystem: Preprocessing, Deskewing, Illumination, and Segmentation.
"""

from app.cv.scanner import (
    detect_document_contour,
    four_point_transform,
    calculate_skew_angle,
    rotate_image,
    rectify_document_geometry,
)
from app.cv.illumination import (
    remove_non_uniform_lighting,
    enhance_contrast_clahe,
    suppress_shadows_and_denoise,
    adaptive_binarize,
    sauvola_threshold,
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
from app.cv.enhancer import (
    enhance_stroke_sharpness,
    enhance_contrast_adaptive,
    suppress_paper_bleed_through,
    suppress_notebook_grid_and_ruled_lines,
    deskew_and_level_line_crop,
    pad_line_crop,
    generate_tta_variants,
)

__all__ = [
    "detect_document_contour",
    "four_point_transform",
    "calculate_skew_angle",
    "rotate_image",
    "rectify_document_geometry",
    "remove_non_uniform_lighting",
    "enhance_contrast_clahe",
    "suppress_shadows_and_denoise",
    "adaptive_binarize",
    "sauvola_threshold",
    "compute_horizontal_projection_profile",
    "segment_line_intervals",
    "segment_text_lines",
    "render_projection_profile_image",
    "draw_bounding_boxes_overlay",
    "save_pipeline_debug_artifacts",
    "enhance_stroke_sharpness",
    "enhance_contrast_adaptive",
    "suppress_paper_bleed_through",
    "suppress_notebook_grid_and_ruled_lines",
    "deskew_and_level_line_crop",
    "pad_line_crop",
    "generate_tta_variants",
]

