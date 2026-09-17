# ### FILE: app/ml/hybrid_ensemble.py
"""
Hybrid Ensemble Recognition Engine: TrOCR-Base (Transformer) + CRNN (CTC).
Combines attention-based linguistic fluency with strict non-hallucinating CTC character decoding.
When Transformer confidence falls below threshold, CTC arbitration resolves ambiguities.
"""

from typing import List, Tuple, Optional
import numpy as np
import re

from app.config import get_settings
from app.core.logging import get_logger
from app.ml.vocabulary_binder import get_vocabulary_binder
from app.ml.handwriting_confusion import get_handwriting_confusion_corrector

logger = get_logger(__name__)
settings = get_settings()


class ConsensusArbitrator:
    """
    Arbitrates between Transformer (TrOCR) and Connectionist Temporal Classification (CRNN)
    hypotheses to eliminate hallucinations and choose the most linguistically sound transcription.
    """

    def __init__(self) -> None:
        self.binder = get_vocabulary_binder()
        self.confusion = get_handwriting_confusion_corrector()

    def has_repetitive_loop(self, text: str) -> bool:
        """Detect autoregressive decoder repetition loop artifacts (e.g. 'слово слово слово')."""
        words = text.strip().split()
        if len(words) >= 4:
            for i in range(len(words) - 3):
                if words[i].lower() == words[i + 1].lower() == words[i + 2].lower():
                    return True
        # Check repeated character sequences like 'аааааа' or '......'
        if re.search(r"(\S{2,})\1{3,}", text):
            return True
        return False

    def count_valid_russian_words(self, text: str) -> Tuple[int, int]:
        """Count how many space-delimited tokens match known Russian lexicon or valid grammar."""
        tokens = re.findall(r"[А-Яа-яЁё]+", text)
        if not tokens:
            return 0, 0
        valid = sum(1 for t in tokens if len(t) <= 2 or t.lower() in self.confusion.vocabulary)
        return valid, len(tokens)

    def arbitrate(
        self,
        trocr_text: str,
        trocr_conf: float,
        crnn_text: str,
        crnn_conf: float,
    ) -> Tuple[str, float, str]:
        """
        Arbitrate between TrOCR and CRNN predictions.
        Returns (chosen_text, combined_confidence, decision_reason).
        """
        trocr_clean = trocr_text.strip()
        crnn_clean = crnn_text.strip()

        # If one engine returned empty, take the other
        if not trocr_clean and crnn_clean:
            return crnn_clean, crnn_conf, "crnn_fallback_empty_trocr"
        if not crnn_clean and trocr_clean:
            return trocr_clean, trocr_conf, "trocr_only"
        if not trocr_clean and not crnn_clean:
            return "", 0.0, "both_empty"

        # 1. Immediate override if TrOCR fell into a repetitive loop
        if self.has_repetitive_loop(trocr_clean) and not self.has_repetitive_loop(crnn_clean):
            logger.info("Consensus: TrOCR repetitive loop detected ('%s'), overriding with CRNN ('%s')", trocr_clean[:30], crnn_clean[:30])
            return crnn_clean, max(crnn_conf, 0.75), "crnn_override_repetitive_loop"

        # 2. Strong TrOCR confidence
        if trocr_conf >= 0.88:
            return trocr_clean, trocr_conf, "trocr_high_confidence"

        # 3. Check lexical validity (dictionary words)
        t_valid, t_total = self.count_valid_russian_words(trocr_clean)
        c_valid, c_total = self.count_valid_russian_words(crnn_clean)

        t_ratio = t_valid / max(1, t_total)
        c_ratio = c_valid / max(1, c_total)

        # If CRNN has significantly higher valid word ratio on a low-confidence TrOCR
        if c_ratio > t_ratio + 0.35 and c_valid >= 2 and crnn_conf >= 0.65:
            logger.info("Consensus: CRNN has higher lexical validity (%.2f vs %.2f). Picking CRNN.", c_ratio, t_ratio)
            return crnn_clean, (trocr_conf * 0.3 + crnn_conf * 0.7), "crnn_lexical_advantage"

        # 4. If predictions are close or identical
        if trocr_clean.lower() == crnn_clean.lower():
            boosted_conf = min(0.98, max(trocr_conf, crnn_conf) + 0.06)
            return trocr_clean, boosted_conf, "unanimous_agreement"

        # 5. Default weighted ensemble favouring TrOCR for language modeling fluency
        chosen = trocr_clean if (trocr_conf * 0.65 >= crnn_conf * 0.35) else crnn_clean
        combined = trocr_conf * 0.60 + crnn_conf * 0.40
        return chosen, combined, "ensemble_weighted"


class HybridEnsembleEngine:
    """
    High-accuracy HTR Engine orchestrating TrOCR-Base and CRNN with fast-path consensus.
    """

    def __init__(
        self,
        transformer_engine=None,
        crnn_engine=None,
        consensus_threshold: float = 0.88,
    ) -> None:
        self.consensus_threshold = consensus_threshold
        self.arbitrator = ConsensusArbitrator()
        self.binder = get_vocabulary_binder()

        # Lazy load engines if not provided
        if transformer_engine is not None:
            self.transformer = transformer_engine
        else:
            try:
                from app.ml.transformer_engine import TransformerHTREngine
                self.transformer = TransformerHTREngine()
            except Exception as exc:
                logger.warning("TrOCR init in HybridEnsemble failed, running CRNN only: %s", exc)
                self.transformer = None

        if crnn_engine is not None:
            self.crnn = crnn_engine
        else:
            try:
                from app.ml.inference import CRNNInferenceEngine
                self.crnn = CRNNInferenceEngine()
            except Exception as exc:
                logger.warning("CRNN init in HybridEnsemble failed: %s", exc)
                self.crnn = None

        logger.info(
            "HybridEnsembleEngine initialized (Transformer: %s, CRNN: %s, Threshold: %.2f)",
            "Active" if self.transformer else "None",
            "Active" if self.crnn else "None",
            self.consensus_threshold,
        )

    def predict_single_line(
        self,
        crop: np.ndarray,
        enable_tta: bool = True,
        context_prev_line: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Recognize a single line crop using Hybrid TrOCR + CRNN consensus arbitration.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
            return "", 0.0

        # If only one engine is available, use it directly
        if self.transformer is None and self.crnn is not None:
            res = self.crnn.predict_batch([crop], use_beam_search=True)
            return res[0] if res else ("", 0.0)
        if self.crnn is None and self.transformer is not None:
            return self.transformer.predict_single_line(
                crop, enable_tta=enable_tta, context_prev_line=context_prev_line
            )

        # 1. First stage: TrOCR fast-path
        t_text, t_conf = self.transformer.predict_single_line(
            crop, enable_tta=enable_tta, context_prev_line=context_prev_line
        )

        # Early exit on confident TrOCR result without repetitive loop
        if t_conf >= self.consensus_threshold and not self.arbitrator.has_repetitive_loop(t_text):
            final_text = self.binder.correct_line_vocabulary(t_text)
            return final_text, t_conf

        # 2. Second stage: CRNN CTC evaluation for ambiguous or repetitive lines
        try:
            crnn_preds = self.crnn.predict_batch([crop], use_beam_search=True)
            c_text, c_conf = crnn_preds[0] if crnn_preds else ("", 0.0)
        except Exception as exc:
            logger.warning("CRNN secondary prediction failed in ensemble: %s", exc)
            c_text, c_conf = "", 0.0

        # 3. Consensus arbitration
        chosen_text, final_conf, reason = self.arbitrator.arbitrate(
            t_text, t_conf, c_text, c_conf
        )

        # Apply Russian syntax and vocabulary binding
        final_text = self.binder.correct_line_vocabulary(chosen_text)
        return final_text, final_conf

    def predict_batch(
        self,
        images: List[np.ndarray],
        use_beam_search: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Run sequential hybrid ensemble inference across a page's line crops,
        propagating cross-line linguistic context.
        """
        if not images:
            return []

        results: List[Tuple[str, float]] = []
        last_text: Optional[str] = None
        total = len(images)

        for idx, crop in enumerate(images):
            text, conf = self.predict_single_line(
                crop,
                enable_tta=True,
                context_prev_line=last_text,
            )
            if text and text.strip():
                last_text = text.strip()
            results.append((text, conf))
            logger.info(
                "Ensemble recognized line %d/%d (%.1f%%): %s",
                idx + 1,
                total,
                conf * 100,
                (text[:45] + "...") if len(text) > 45 else text,
            )

        return results
