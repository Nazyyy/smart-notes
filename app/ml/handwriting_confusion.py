# ### FILE: app/ml/handwriting_confusion.py
"""
Handwriting Confusion Matrix and Optical Levenshtein Corrector.
Models Russian cursive letter stroke ambiguities (e.g. п/и/н/т/м/л, о/а/ю, ш/т/щ/ц, д/у/р/з)
and provides context-aware candidate generation for low-confidence words.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import re
from collections import defaultdict
from app.ml.vocabulary_binder import DomainVocabularyBinder
from app.core.logging import get_logger

logger = get_logger(__name__)


class HandwritingConfusionCorrector:
    """
    Optical confusion-weighted spelling and candidate correction engine.
    Uses domain-specific optical similarity costs for Russian handwriting.
    """

    # Equivalence clusters for Russian handwritten strokes
    OPTICAL_CLUSTERS = [
        # Vertical downstrokes & ligatures (very high confusion in cursive)
        {"п", "и", "н", "т", "м", "л"},
        # Oval loops
        {"о", "а", "ю", "е"},
        # Triple/quadruple vertical loops and hooks
        {"ш", "т", "щ", "ц"},
        # Descenders and lower loops
        {"д", "у", "р", "з"},
        # Ascenders and rounded loops
        {"в", "б", "ъ", "ь", "ы"},
        # Crossed strokes
        {"х", "ж", "к"},
        # Sibilants & rounded hooks
        {"с", "е", "э"},
    ]

    def __init__(self, custom_vocab: Optional[Set[str]] = None) -> None:
        self.confusion_matrix: Dict[Tuple[str, str], float] = {}
        self._build_confusion_weights()
        self.vocabulary: Set[str] = set()
        self._populate_vocabulary(custom_vocab)

    def _build_confusion_weights(self) -> None:
        """Precompute substitution penalties between visually similar Russian cursive letters."""
        for cluster in self.OPTICAL_CLUSTERS:
            for char_a in cluster:
                for char_b in cluster:
                    if char_a != char_b:
                        # Low penalty for visually ambiguous cursive letters
                        self.confusion_matrix[(char_a, char_b)] = 0.20
                        self.confusion_matrix[(char_b, char_a)] = 0.20

    def _populate_vocabulary(self, custom_vocab: Optional[Set[str]] = None) -> None:
        """Populate Russian educational, scientific, and academic vocabulary."""
        # 1. Base academic & high-frequency educational terms
        base_words = [
            "антропогенез", "социогенез", "антропосоциогенез", "биосоциальное", "существо",
            "человек", "человека", "человеческого", "энгельс", "общество", "общества",
            "биологическая", "социальная", "анатомия", "физиология", "сознание", "разум",
            "свобода", "ответственность", "индивид", "личность", "деятельность",
            "мировоззрение", "познание", "истина", "критерии", "практика", "логика",
            "структура", "институты", "нормы", "права", "морали", "развитие", "живых",
            "организмов", "труд", "создал", "способность", "общественно-полезному",
            "прилагательное", "существительное", "суффикс", "соломенный", "огненный",
            "стеклянный", "оловянный", "деревянный", "краткое", "полное", "правило",
            "внутривенно", "струйно", "болюсное", "инфузионное", "капельно", "абсорбция",
            "всасывание", "мембрана", "фильтрация", "диффузия", "пиноцитоз", "грелку",
            "фармакокинетика", "дозировка", "вакуоль", "лимфатическую", "раствора",
            "кислота", "серная", "соляная", "гидроксид", "реакция", "нейтрализация",
            "окисление", "восстановление", "осадок", "электролиз", "молекула", "валентность",
            "интеграл", "производная", "дифференциал", "функция", "предел", "вектор",
            "уравнение", "неравенство", "матрица", "теорема", "коши", "лагранжа",
            "классная", "домашняя", "работа", "упражнение", "задача", "вариант", "решение", "ответ"
        ]
        for w in base_words:
            self.vocabulary.add(w.lower())

        # 2. Add domain binder corrections target words
        for _, canonical in DomainVocabularyBinder.PHRASE_MAPPINGS:
            for w in canonical.split():
                clean_w = re.sub(r'[^а-яА-Яa-zA-Z0-9]', '', w).lower()
                if len(clean_w) > 2:
                    self.vocabulary.add(clean_w)

        if custom_vocab:
            self.vocabulary.update(w.lower() for w in custom_vocab)

    def substitution_cost(self, a: str, b: str) -> float:
        """Return substitution penalty between two characters."""
        if a == b:
            return 0.0
        # Optical confusion pair
        return self.confusion_matrix.get((a.lower(), b.lower()), 1.0)

    def weighted_levenshtein(self, s1: str, s2: str, max_dist: float = 3.0) -> float:
        """
        Compute dynamic programming edit distance with optical handwriting weights.
        Early exits if distance exceeds max_dist.
        """
        m, n = len(s1), len(s2)
        if abs(m - n) > max_dist:
            return float("inf")

        dp = [float(j) for j in range(n + 1)]

        for i in range(1, m + 1):
            new_dp = [float(i)] + [0.0] * n
            c1 = s1[i - 1]
            min_row = new_dp[0]

            for j in range(1, n + 1):
                c2 = s2[j - 1]
                cost = self.substitution_cost(c1, c2)

                new_dp[j] = min(
                    dp[j] + 1.0,          # Deletion
                    new_dp[j - 1] + 1.0,  # Insertion
                    dp[j - 1] + cost      # Substitution (optical-weighted)
                )
                min_row = min(min_row, new_dp[j])

            if min_row > max_dist:
                return float("inf")

            dp = new_dp

        return dp[n]

    def get_word_candidates(
        self,
        word: str,
        context_words: Optional[List[str]] = None,
        top_k: int = 3,
        max_dist: float = 2.2,
    ) -> List[Dict[str, Any]]:
        """
        Search vocabulary for most likely correct alternatives for a misrecognized or uncertain word.
        Returns list of dicts: [{'word': cand, 'score': score, 'dist': dist, 'reason': reason}]
        """
        clean_word = re.sub(r'[^а-яА-Яa-zA-Z0-9]', '', word).lower()
        if not clean_word or len(clean_word) < 3:
            return []

        # If word is already completely valid and canonical, return it with 1.0 score
        if clean_word in self.vocabulary:
            return [{"word": clean_word, "score": 1.0, "dist": 0.0, "reason": "Словарное слово"}]

        candidates: List[Tuple[float, float, str, str]] = []

        w_len = len(clean_word)
        for vocab_word in self.vocabulary:
            if abs(len(vocab_word) - w_len) > 2:
                continue

            dist = self.weighted_levenshtein(clean_word, vocab_word, max_dist=max_dist)
            if dist <= max_dist:
                # Contextual bonus if vocab_word matches topic
                ctx_bonus = 0.0
                if context_words:
                    for cw in context_words:
                        if cw and cw.lower() in self.vocabulary:
                            ctx_bonus += 0.05

                similarity_score = max(0.0, 1.0 - (dist / max(1.0, float(len(vocab_word))))) + ctx_bonus
                reason = "Оптическая замена букв" if dist < 1.0 else "Похожее словарное слово"
                candidates.append((similarity_score, -dist, vocab_word, reason))

        candidates.sort(reverse=True)
        results: List[Dict[str, Any]] = []
        seen = set()

        for score, neg_dist, cand, reason in candidates:
            if cand not in seen:
                seen.add(cand)
                # Preserve capitalization of original word if capitalized
                if word and word[0].isupper():
                    cand_display = cand.capitalize()
                else:
                    cand_display = cand

                results.append({
                    "word": cand_display,
                    "score": round(score, 3),
                    "dist": round(-neg_dist, 2),
                    "reason": reason,
                })
                if len(results) >= top_k:
                    break

        return results


_singleton_confusion_corrector: Optional[HandwritingConfusionCorrector] = None


def get_handwriting_confusion_corrector() -> HandwritingConfusionCorrector:
    """Singleton getter for HandwritingConfusionCorrector."""
    global _singleton_confusion_corrector
    if _singleton_confusion_corrector is None:
        _singleton_confusion_corrector = HandwritingConfusionCorrector()
    return _singleton_confusion_corrector
