# ### FILE: app/ml/transformer_engine.py
"""
SOTA Transformer-based Handwritten Text Recognition Engine (TrOCR).
Utilizes Vision Transformer (ViT) Encoder + RoBERTa Decoder fine-tuned
for Cyrillic, Latin, and scientific handwritten text recognition.
Features intelligent multi-column span segmentation and contextual academic post-processing.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import re
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from app.config import get_settings
from app.core.exceptions import ModelInferenceException
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


from app.ml.vocabulary_binder import DomainVocabularyBinder


def postprocess_scientific_and_academic(text: str) -> str:
    """
    Domain-specific normalizer for academic, scientific/chemistry, and common OCR artifacts.
    Standardizes chemical formulas, reactions, and school/university terminology.
    """
    if not text:
        return ""

    # Split multi-column segments if present
    parts = [p.strip() for p in text.split("  |  ")]
    cleaned_parts: List[str] = []
    for p in parts:
        # Filter out pure punctuation or single-letter noise artifacts (e.g. '.', 'М.', 'О.', 'X')
        alpha_chars = re.sub(r"[\s\.\,\-\:\;\"\'\!\?\(\)]", "", p)
        if len(alpha_chars) <= 1 and not p.strip().isdigit():
            continue
        cleaned_parts.append(p)

    if not cleaned_parts:
        s = text
    else:
        s = " | ".join(cleaned_parts)

    # Apply comprehensive domain vocabulary binding and error correction
    s = DomainVocabularyBinder.clean_and_bind(s)
    return s


class TransformerHTREngine:
    """
    Production Transformer HTR Engine using TrOCR.
    Processes line crops through aspect-ratio preserved padding and intelligent
    multi-column span segmentation, yielding high-accuracy Cyrillic and scientific transcriptions.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
        batch_size: int = 8,
    ) -> None:
        preferred_path = Path(model_path or settings.ML_TRANSFORMER_PATH)
        # Check if line-level model exists, otherwise fallback to base model
        if not preferred_path.exists():
            alt_path = Path("./data/weights/trocr_ru_lines")
            if alt_path.exists():
                preferred_path = alt_path
            else:
                preferred_path = Path("./data/weights/trocr_ru")

        self.model_path = preferred_path
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() and settings.ML_DEVICE == "cuda" else "cpu")
        )
        self.batch_size = batch_size

        logger.info("Initializing TransformerHTREngine from %s on %s...", self.model_path, self.device)
        try:
            self.processor = TrOCRProcessor.from_pretrained(str(self.model_path))
            self.model = VisionEncoderDecoderModel.from_pretrained(str(self.model_path))
            self.model.to(self.device)
            self.model.eval()
            logger.info("TransformerHTREngine loaded successfully on %s.", self.device)
        except Exception as exc:
            logger.error("Failed to load Transformer model from %s: %s", self.model_path, exc)
            raise ModelInferenceException(f"Failed to load Transformer model: {exc}") from exc

    def _segment_line_into_spans(self, crop: np.ndarray) -> List[Tuple[np.ndarray, bool]]:
        """
        Intelligently detect large column gaps (>= 38px of white space) such as
        two-column notebook tables or date headers.
        Returns a list of (sub_crop, is_split) tuples. Trims empty margin on the right.
        """
        h, w = crop.shape[:2]
        if w <= 100:
            return [(crop, False)]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        _, bin_crop = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        vpp = np.sum(bin_crop > 0, axis=0)

        # Detect wide zero/near-zero ink gaps (at least 38px, not at image boundaries)
        is_gap = vpp < 2
        split_xs: List[int] = []
        in_gap = False
        gap_start = 0

        for x, g in enumerate(is_gap):
            if g and not in_gap:
                in_gap = True
                gap_start = x
            elif not g and in_gap:
                in_gap = False
                gap_len = x - gap_start
                if gap_len >= 38 and gap_start >= 40:
                    ink_left = int(np.sum(bin_crop[:, :gap_start] > 0))
                    ink_right = int(np.sum(bin_crop[:, x:] > 0))
                    # Only split as column if both sides have real text
                    if (w - x) >= 40 and ink_left >= 60 and ink_right >= 60:
                        split_xs.append((gap_start + x) // 2)
                    elif ink_right < 60:
                        # Right side is just empty margin paper, trim it!
                        crop = crop[:, :gap_start]
                        break

        if not split_xs:
            return [(crop, False)]

        # Cut column spans
        bounds = [0] + split_xs + [crop.shape[1]]
        spans: List[Tuple[np.ndarray, bool]] = []
        for b1, b2 in zip(bounds[:-1], bounds[1:]):
            sub_crop = crop[:, b1:b2]
            if sub_crop.shape[1] >= 16:
                spans.append((sub_crop, True))

        return spans if spans else [(crop, False)]

    def _prepare_pil_crop(self, crop: np.ndarray) -> Image.Image:
        """
        Pad crop with a small white border to preserve outer letter strokes
        when scaled to TrOCR's 384x384 input resolution.
        """
        h, w = crop.shape[:2]
        pad_top = max(4, int(h * 0.08))
        pad_side = max(10, int(w * 0.03))
        padded = cv2.copyMakeBorder(
            crop, pad_top, pad_top, pad_side, pad_side, cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        if len(padded.shape) == 2:
            rgb = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
        elif padded.shape[2] == 4:
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def predict_single_line(self, crop: np.ndarray) -> Tuple[str, float]:
        """Recognize a single line crop using intelligent span segmentation and TrOCR."""
        if crop is None or crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
            return "", 0.0

        spans = self._segment_line_into_spans(crop)
        pil_images = [self._prepare_pil_crop(c) for c, _ in spans]

        pixel_values = self.processor(images=pil_images, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                max_new_tokens=64,
                num_beams=3,
                return_dict_in_generate=True,
                output_scores=True,
            )

        token_ids = outputs.sequences
        span_texts = self.processor.batch_decode(token_ids, skip_special_tokens=True)

        # Confidence calculation
        conf = 0.88
        try:
            scores = self.model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
            probs = torch.exp(scores)
            valid_probs = probs[probs > 0]
            if len(valid_probs) > 0:
                conf = float(valid_probs.mean().item())
                conf = max(0.25, min(0.99, conf))
        except Exception:
            conf = 0.88

        # If line was split across columns (tables), join with column separator
        is_table = any(is_split for _, is_split in spans)
        if is_table and len(span_texts) > 1:
            raw_text = "  |  ".join(span_texts)
        else:
            raw_text = " ".join(span_texts)

        clean_text = postprocess_scientific_and_academic(raw_text)
        return clean_text, round(conf, 4)

    def predict_batch(
        self,
        images: List[np.ndarray],
        use_beam_search: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Run inference across a batch of line crops.
        """
        if not images:
            return []

        results: List[Tuple[str, float]] = []
        try:
            for crop in images:
                text, conf = self.predict_single_line(crop)
                results.append((text, conf))
        except Exception as exc:
            logger.error("Transformer batch prediction failed: %s", exc, exc_info=True)
            raise ModelInferenceException(f"Transformer batch prediction failed: {exc}") from exc
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return results
