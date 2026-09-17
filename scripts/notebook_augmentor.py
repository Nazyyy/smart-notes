# ### FILE: scripts/notebook_augmentor.py
"""
Handwritten Notebook Augmentor.
Synthesizes notebook ruled/grid backgrounds, ink variations, shear, and physical camera blur.
Prepares robust training crops for TrOCR on real Russian school and university notebooks.
"""

from typing import Tuple, Optional
import random
import numpy as np
import cv2


class HandwrittenNotebookAugmentor:
    """
    Advanced physical and optical augmentations for handwritten notebook lines.
    """

    @staticmethod
    def add_notebook_grid(image: np.ndarray, prob: float = 0.5) -> np.ndarray:
        """Add realistic checkered notebook grid lines under or over ink."""
        if random.random() > prob:
            return image

        h, w = image.shape[:2]
        canvas = image.copy()

        # Grid spacing: 20 to 38 pixels (typical standard 5mm notebook cell)
        grid_step = random.randint(22, 38)
        grid_color = (
            random.randint(180, 225),  # B
            random.randint(190, 230),  # G
            random.randint(190, 235),  # R
        )
        alpha = random.uniform(0.18, 0.42)

        overlay = canvas.copy()

        # Horizontal grid lines
        y_start = random.randint(0, grid_step)
        for y in range(y_start, h, grid_step):
            cv2.line(overlay, (0, y), (w, y), grid_color, thickness=1)

        # Vertical grid lines
        x_start = random.randint(0, grid_step)
        for x in range(x_start, w, grid_step):
            cv2.line(overlay, (x, 0), (x, h), grid_color, thickness=1)

        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
        return canvas

    @staticmethod
    def add_ruled_lines(image: np.ndarray, prob: float = 0.35) -> np.ndarray:
        """Add horizontal ruled notebook lines (линейка)."""
        if random.random() > prob:
            return image

        h, w = image.shape[:2]
        canvas = image.copy()
        step = random.randint(28, 48)
        line_color = (
            random.randint(170, 215),
            random.randint(180, 220),
            random.randint(200, 235),
        )
        alpha = random.uniform(0.22, 0.45)
        overlay = canvas.copy()

        y_start = random.randint(0, step)
        for y in range(y_start, h, step):
            cv2.line(overlay, (0, y), (w, y), line_color, thickness=1)

        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
        return canvas

    @staticmethod
    def apply_shear(image: np.ndarray, prob: float = 0.45) -> np.ndarray:
        """Simulate natural handwriting slant/italic variations."""
        if random.random() > prob:
            return image

        h, w = image.shape[:2]
        shear_factor = random.uniform(-0.18, 0.18)
        M = np.float32([[1, shear_factor, 0], [0, 1, 0]])

        # Paper background fill
        mean_bg = tuple(int(c) for c in np.mean(image[:3, :3], axis=(0, 1))) if image.ndim == 3 else 255
        sheared = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=mean_bg)
        return sheared

    @staticmethod
    def apply_motion_blur(image: np.ndarray, prob: float = 0.35) -> np.ndarray:
        """Simulate slight camera shake or hand motion blur during notebook photography."""
        if random.random() > prob:
            return image

        k_size = random.choice([3, 5])
        kernel = np.zeros((k_size, k_size))
        if random.random() > 0.5:
            # Horizontal motion blur
            kernel[int((k_size - 1) / 2), :] = np.ones(k_size)
        else:
            # Diagonal motion blur
            np.fill_diagonal(kernel, 1)
        kernel /= k_size
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def apply_ink_degradation(image: np.ndarray, prob: float = 0.3) -> np.ndarray:
        """Simulate faint pen pressure or dry ballpoint pen strokes."""
        if random.random() > prob:
            return image

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        if random.random() > 0.5:
            # Thin faint strokes
            return cv2.dilate(image, kernel, iterations=1)
        else:
            # Heavy bleed / thick gel pen
            return cv2.erode(image, kernel, iterations=1)

    @classmethod
    def augment(cls, image: np.ndarray) -> np.ndarray:
        """Run full augmentation pipeline on a line crop."""
        aug = image.copy()
        aug = cls.add_notebook_grid(aug, prob=0.45)
        aug = cls.add_ruled_lines(aug, prob=0.30)
        aug = cls.apply_shear(aug, prob=0.40)
        aug = cls.apply_motion_blur(aug, prob=0.30)
        aug = cls.apply_ink_degradation(aug, prob=0.25)
        return aug
