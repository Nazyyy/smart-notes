# ### FILE: app/ml/inference.py
"""
PyTorch Inference Engine for Handwritten Text Recognition.
Handles batch tensor preparation, forward pass execution, and CTC decoding.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
import cv2

from app.config import get_settings
from app.core.exceptions import ModelInferenceException
from app.core.logging import get_logger
from app.ml.crnn_model import CRNN
from app.ml.decoder import CTCGreedyDecoder, CTCBeamSearchDecoder
from app.ml.vocab import VOCAB
from app.ml.weights_initializer import ensure_weights_exist

logger = get_logger(__name__)
settings = get_settings()


class CRNNInferenceEngine:
    """Production inference engine managing CRNN weights lifecycle and execution."""

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        beam_width: Optional[int] = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() and settings.ML_DEVICE == "cuda" else "cpu")
        )
        self.batch_size = batch_size or settings.ML_BATCH_SIZE
        self.target_height = settings.ML_IMAGE_HEIGHT
        self.target_width = settings.ML_IMAGE_WIDTH

        weights_file = weights_path or settings.ML_MODEL_WEIGHTS_PATH
        actual_weights_path = ensure_weights_exist(weights_file)

        self.model = CRNN(
            img_channel=1,
            num_classes=len(VOCAB),
            rnn_hidden_size=256,
            dropout_prob=0.0,
        )

        self._load_weights(actual_weights_path)
        self.model.to(self.device)
        self.model.eval()

        self.greedy_decoder = CTCGreedyDecoder(vocab=VOCAB, blank_idx=settings.ML_BLANK_INDEX)
        self.beam_decoder = CTCBeamSearchDecoder(
            vocab=VOCAB,
            beam_width=beam_width or settings.ML_BEAM_WIDTH,
            blank_idx=settings.ML_BLANK_INDEX,
        )
        logger.info("CRNN Inference Engine initialized successfully on %s", self.device)

    def _load_weights(self, weights_path: Path) -> None:
        """Load checkpoint weights into model state dict."""
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)
            else:
                raise ValueError("Unrecognized checkpoint format.")
        except Exception as exc:
            logger.error("Failed loading model weights from %s: %s", weights_path, exc)
            raise ModelInferenceException(
                reason=f"Failed loading weights: {exc}",
                details={"weights_path": str(weights_path)},
            ) from exc

    def preprocess_line_image(
        self,
        line_img: np.ndarray,
        target_width: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Preprocess a cropped single-line numpy image:
        1. Grayscale conversion.
        2. Proportional resize to fixed height (32px), preserving aspect ratio.
        3. Dynamic canvas padding with 255 (white) up to target_width or scaled width.
        4. Normalization to [-1.0, 1.0].
        Returns tensor of shape (1, 1, 32, W).
        """
        if line_img is None or line_img.size == 0:
            raise ModelInferenceException("Input line image is empty or None.")

        # Ensure grayscale
        if len(line_img.shape) == 3:
            gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_img.copy()

        h, w = gray.shape
        if h <= 0 or w <= 0:
            raise ModelInferenceException(f"Invalid crop dimensions: h={h}, w={w}")

        # Compute scaling factor preserving aspect ratio
        scale = self.target_height / float(h)
        scaled_w = max(32, int(round(w * scale / 4.0)) * 4)

        # Allow dynamic width up to 1280px to prevent squashing multi-word lines
        canvas_w = target_width or max(scaled_w, self.target_width)
        canvas_w = min(canvas_w, 1280)
        canvas_w = max(32, (canvas_w // 4) * 4)

        new_w = min(scaled_w, canvas_w)
        resized = cv2.resize(gray, (new_w, self.target_height), interpolation=cv2.INTER_AREA)

        # Create padded canvas filled with 255 (background)
        canvas = np.full((self.target_height, canvas_w), 255, dtype=np.uint8)
        canvas[:, :new_w] = resized

        # Normalize to [-1.0, 1.0]
        norm = (canvas.astype(np.float32) / 127.5) - 1.0

        # Expand to (1, 1, H, W)
        tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)
        return tensor

    def predict_batch(
        self,
        images: List[np.ndarray],
        use_beam_search: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Run batched inference over a list of cropped line images.
        Returns list of (recognized_text, confidence) tuples.
        """
        if not images:
            return []

        results: List[Tuple[str, float]] = []

        try:
            with torch.no_grad():
                for i in range(0, len(images), self.batch_size):
                    batch_slice = images[i : i + self.batch_size]

                    # Compute optimal batch canvas width preserving aspect ratio across slice
                    slice_scaled_widths = [
                        max(32, int(round((self.target_height / float(max(1, img.shape[0]))) * img.shape[1] / 4.0)) * 4)
                        for img in batch_slice
                        if img is not None and img.size > 0
                    ]
                    batch_w = max(self.target_width, max(slice_scaled_widths)) if slice_scaled_widths else self.target_width
                    batch_w = min(batch_w, 1280)
                    batch_w = max(32, (batch_w // 4) * 4)

                    tensors = [self.preprocess_line_image(img, target_width=batch_w) for img in batch_slice]
                    batch_tensor = torch.cat(tensors, dim=0).to(self.device)

                    # Forward pass -> (T, B, num_classes)
                    log_probs = self.model(batch_tensor)

                    # Decode
                    if use_beam_search:
                        batch_decoded = self.beam_decoder.decode(log_probs)
                    else:
                        batch_decoded = self.greedy_decoder.decode(log_probs)

                    results.extend(batch_decoded)

        except Exception as exc:
            logger.error("Inference execution failed: %s", exc, exc_info=True)
            raise ModelInferenceException(
                reason=f"Model forward pass or decode failed: {exc}"
            ) from exc
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return results
