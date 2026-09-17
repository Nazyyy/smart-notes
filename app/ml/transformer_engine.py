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


from app.ml.vocabulary_binder import DomainVocabularyBinder, postprocess_scientific_and_academic
from app.ml.language_model_rescorer import get_language_model_rescorer
from app.cv.enhancer import generate_tta_variants
from app.cv.segmentation import suppress_grid_lines


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
        # Filter out pure punctuation or marginal abbreviations (e.g. '.', 'М.', 'О.', 'Ул.', 'X')
        clean_part = re.sub(r'\b([ЗзМмОоXxХхУуВвЕеКкРрПпСсТтИиЮюФф]\.|\bУ[лл]\.?)\b', '', p).strip()
        alpha_chars = re.sub(r"[\s\.\,\-\:\;\"\'\!\?\(\)]", "", clean_part)
        if len(alpha_chars) <= 2 and not clean_part.isdigit():
            continue
        cleaned_parts.append(clean_part)

    if not cleaned_parts:
        s = text
    else:
        s = "  |  ".join(cleaned_parts)

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
        # Check if fine-tuned or line-level model exists, otherwise fallback to base model
        if not preferred_path.exists():
            for alt_candidate in [
                Path("./data/weights/trocr_base_academic"),
                Path("./data/weights/trocr_finetuned_academic"),
                Path("./data/weights/trocr_ru_lines"),
                Path("./data/weights/trocr_ru"),
            ]:
                if alt_candidate.exists():
                    preferred_path = alt_candidate
                    break

        self.model_path = preferred_path
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() and settings.ML_DEVICE == "cuda" else "cpu")
        )
        self.batch_size = batch_size

        logger.info("Initializing TransformerHTREngine from %s on %s...", self.model_path, self.device)
        try:
            self.processor = TrOCRProcessor.from_pretrained(str(self.model_path))
            self.model = VisionEncoderDecoderModel.from_pretrained(str(self.model_path))
            try:
                self.model.to(self.device)
            except torch.cuda.OutOfMemoryError:
                logger.warning("CUDA memory pressure encountered, falling back to CPU for inference.")
                self.device = torch.device("cpu")
                self.model.to(self.device)

            self.model.eval()

            if self.device.type == "cuda":
                try:
                    torch.backends.cuda.enable_flash_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                except Exception:
                    pass

            logger.info("TransformerHTREngine loaded successfully on %s.", self.device)

        except Exception as exc:
            logger.error("Failed to load Transformer model from %s: %s", self.model_path, exc)
            raise ModelInferenceException(f"Failed to load Transformer model: {exc}") from exc

    def _segment_line_into_spans(self, crop: np.ndarray) -> List[Tuple[np.ndarray, bool]]:
        """
        Intelligently detect large table column gaps (>= 75px or 12% width of white space).
        Prevents splitting normal sentences into fractured long and short segments.
        Trims empty paper margins on the left and right.
        """
        h, w = crop.shape[:2]
        if w <= 140:
            return [(crop, False)]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        _, bin_crop = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        clean_crop = suppress_grid_lines(bin_crop)
        vpp = np.sum(clean_crop > 0, axis=0)

        # Minimum gap length and side margins to qualify as genuine multi-column layout
        min_gap = max(75, int(w * 0.12))
        min_side_w = max(80, int(w * 0.12))

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
                if gap_len >= min_gap and gap_start >= min_side_w:
                    ink_left = int(np.sum(clean_crop[:, :gap_start] > 0))
                    ink_right = int(np.sum(clean_crop[:, x:] > 0))
                    # Only split as column if BOTH sides are genuine substantial text blocks
                    if (w - x) >= min_side_w and ink_left >= 200 and ink_right >= 200:
                        split_xs.append((gap_start + x) // 2)
                    elif ink_right < 60:
                        # Right side is just empty paper margin, trim it!
                        crop = crop[:, :gap_start]
                        break
                    elif ink_left < 60:
                        # Left side is empty paper margin, trim it!
                        crop = crop[:, x:]
                        break

        if not split_xs:
            return [(crop, False)]

        # Cut column spans
        bounds = [0] + split_xs + [crop.shape[1]]
        spans: List[Tuple[np.ndarray, bool]] = []
        for b1, b2 in zip(bounds[:-1], bounds[1:]):
            sub_crop = crop[:, b1:b2]
            if sub_crop.shape[1] >= 25:
                # verify sub_crop has actual ink and contrast
                g_sub = cv2.cvtColor(sub_crop, cv2.COLOR_BGR2GRAY) if len(sub_crop.shape) == 3 else sub_crop
                _, b_sub = cv2.threshold(g_sub, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                c_sub = suppress_grid_lines(b_sub)
                if np.sum(c_sub > 0) >= 80 and float(np.std(g_sub)) >= 13.0:
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

    def predict_single_line(
        self,
        crop: np.ndarray,
        enable_tta: bool = True,
        context_prev_line: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Recognize a single line crop using multi-column span detection,
        Fast-Path Test-Time Augmentation (TTA) with early exit, cross-line context,
        and N-gram beam rescoring.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
            return "", 0.0

        spans = self._segment_line_into_spans(crop)
        span_results: List[Tuple[str, float]] = []
        rescorer = get_language_model_rescorer()

        for sub_crop, _ in spans:
            if sub_crop is None or sub_crop.size == 0 or sub_crop.shape[1] < 12:
                continue

            # --- Fast-Path: Evaluate primary unaugmented crop first ---
            pil_primary = self._prepare_pil_crop(sub_crop)
            pixel_primary = self.processor(images=[pil_primary], return_tensors="pt").pixel_values.to(self.device)

            with torch.no_grad():
                if self.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model.generate(
                            pixel_primary,
                            max_new_tokens=64,
                            num_beams=5,
                            repetition_penalty=1.25,
                            no_repeat_ngram_size=3,
                            early_stopping=True,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                else:
                    outputs = self.model.generate(
                        pixel_primary,
                        max_new_tokens=64,
                        num_beams=5,
                        repetition_penalty=1.25,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

            try:
                scores = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
                )
                probs = torch.exp(scores[0])
                valid_probs = probs[probs > 0]
                primary_conf = float(valid_probs.mean().item()) if len(valid_probs) > 0 else 0.85
            except Exception:
                primary_conf = 0.85

            primary_text = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            bound_primary = postprocess_scientific_and_academic(primary_text)

            # Fast-path early exit: if primary crop has >= 90% confidence, skip remaining TTA variants!
            if enable_tta and primary_conf >= 0.90 and bound_primary.strip():
                span_results.append((bound_primary, primary_conf))
                continue

            # If TTA enabled and confidence < 90%, evaluate multi-view variants
            if enable_tta and sub_crop.shape[1] >= 24:
                variants = generate_tta_variants(sub_crop)
                # Skip variant 0 as we already processed it
                extra_variants = variants[1:] if len(variants) > 1 else []
                if extra_variants:
                    pil_extras = [self._prepare_pil_crop(v[1]) for v in extra_variants]
                    pixel_extras = self.processor(images=pil_extras, return_tensors="pt").pixel_values.to(self.device)

                    with torch.no_grad():
                        if self.device.type == "cuda":
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                extra_outputs = self.model.generate(
                                    pixel_extras,
                                    max_new_tokens=64,
                                    num_beams=3,
                                    repetition_penalty=1.25,
                                    no_repeat_ngram_size=3,
                                    early_stopping=True,
                                    return_dict_in_generate=True,
                                    output_scores=True,
                                )
                        else:
                            extra_outputs = self.model.generate(
                                pixel_extras,
                                max_new_tokens=64,
                                num_beams=3,
                                repetition_penalty=1.25,
                                no_repeat_ngram_size=3,
                                early_stopping=True,
                                return_dict_in_generate=True,
                                output_scores=True,
                            )

                    try:
                        extra_scores = self.model.compute_transition_scores(
                            extra_outputs.sequences, extra_outputs.scores, extra_outputs.beam_indices, normalize_logits=True
                        )
                    except Exception:
                        extra_scores = None

                    extra_texts = self.processor.batch_decode(extra_outputs.sequences, skip_special_tokens=True)
                else:
                    extra_texts = []
                    extra_scores = None

                candidates: dict[str, list[float]] = {}
                if bound_primary:
                    candidates[bound_primary] = [primary_conf]

                for e_idx, e_txt in enumerate(extra_texts):
                    e_conf = 0.82
                    if extra_scores is not None:
                        e_probs = torch.exp(extra_scores[e_idx])
                        e_val = e_probs[e_probs > 0]
                        if len(e_val) > 0:
                            e_conf = float(e_val.mean().item())

                    e_bound = postprocess_scientific_and_academic(e_txt)
                    if not e_bound:
                        continue
                    if e_bound not in candidates:
                        candidates[e_bound] = []
                    candidates[e_bound].append(e_conf)

                if candidates:
                    scored = []
                    for c_text, conf_list in candidates.items():
                        words = c_text.split()
                        length_weight = 1.0 if len(words) >= 3 else (0.75 if len(words) == 2 else 0.4)
                        consensus_bonus = 0.15 * (len(conf_list) - 1)
                        mean_conf = float(np.mean(conf_list))
                        lm_score = rescorer.score_sequence(c_text, prev_context=context_prev_line)
                        total_score = (mean_conf + consensus_bonus + 0.04 * lm_score) * length_weight
                        scored.append((total_score, mean_conf, c_text))
                    scored.sort(reverse=True)
                    span_results.append((scored[0][2], scored[0][1]))
                else:
                    span_results.append(("", 0.0))
            else:
                span_results.append((bound_primary, primary_conf))

        if not span_results:
            return "", 0.0

        # If line was split across columns (tables), join with column separator
        is_table = any(is_split for _, is_split in spans)
        valid_spans = [t for t, _ in span_results if t.strip()]
        if not valid_spans:
            return "", 0.0

        if is_table and len(valid_spans) > 1:
            line_text = "  |  ".join(valid_spans)
        else:
            line_text = " ".join(valid_spans)

        avg_conf = float(np.mean([c for _, c in span_results if c > 0])) if span_results else 0.88
        return line_text, round(avg_conf, 4)

    def predict_batch(
        self,
        images: List[np.ndarray],
        use_beam_search: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Run sequential inference across a batch of line crops,
        propagating cross-line linguistic context from each recognized line to the next.
        """
        if not images:
            return []

        results: List[Tuple[str, float]] = []
        last_recognized_text: Optional[str] = None

        try:
            for crop in images:
                text, conf = self.predict_single_line(
                    crop,
                    enable_tta=True,
                    context_prev_line=last_recognized_text,
                )
                if text and text.strip():
                    last_recognized_text = text.strip()
                results.append((text, conf))
        except Exception as exc:
            logger.error("Transformer batch prediction failed: %s", exc, exc_info=True)
            raise ModelInferenceException(f"Transformer batch prediction failed: {exc}") from exc
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return results
