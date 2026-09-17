# ### FILE: app/ml/language_model_rescorer.py
"""
Statistical N-Gram Language Model and Beam Search Rescorer.
Evaluates lexical and n-gram perplexity of candidate OCR sequences,
penalizing invalid pseudo-words and promoting canonical Russian educational phrases.
"""

from typing import List, Tuple, Dict, Optional, Set
import math
import re
from collections import Counter, defaultdict
from app.core.logging import get_logger

logger = get_logger(__name__)


class NGramLanguageModelRescorer:
    """
    Lightweight, highly optimized N-gram Language Model for beam search rescoring.
    Combines word-level and subword character-level statistics with Witten-Bell smoothing.
    """

    def __init__(self, vocabulary: Optional[Set[str]] = None):
        self.unigrams: Counter = Counter()
        self.bigrams: Counter = Counter()
        self.trigrams: Counter = Counter()
        self.char_trigrams: Counter = Counter()
        self.total_unigrams: int = 0
        self.total_bigrams: int = 0
        self.total_trigrams: int = 0
        self.total_char_trigrams: int = 0
        self.vocab: Set[str] = vocabulary if vocabulary else set()

        self._build_default_educational_corpus()

    def _build_default_educational_corpus(self) -> None:
        """Seed language model with academic, scientific, and educational texts."""
        seed_corpus = [
            # Social studies & philosophy
            "антропогенез теория возникновения человека и общества",
            "социогенез теория становления и развитие человеческого общества",
            "антропосоциогенез происхождение и развитие человека и общества",
            "человек биосоциальное существо высшая ступень развития живых организмов",
            "человека создал труд способность к общественно-полезному труду",
            "биологическая и социальная природа человека анатомия физиология",
            "сознание и разум свобода и ответственность индивида в обществе",
            "структура общества социальные институты и нормы права морали",
            "познание мира чувственное и рациональное познание истины",
            "научное мировоззрение критерии истины практика и логика",

            # Medicine, Biology & Pharmacology
            "не более 10 мл раствора для инъекций",
            "если ускорить кладем грелку замедлить пузырь со льдом",
            "внутривенно болюсное струйно инфузионное капельно комбинированное",
            "абсорбция всасывание процесс поступления лекарственного средства",
            "места введения в кровеносную и лимфатическую систему через биологические мембраны",
            "основные пути всасывания пассивная диффузия по градиенту концентрации",
            "фильтрация процесс поступления вещества через поры в мембране",
            "пиноцитоз процесс проникновения через мембрану с образованием вакуоли",
            "фармакокинетика и фармакодинамика лекарственных препаратов",
            "биодоступность период полувыведения клиренс и метаболизм",
            "строение клетки ядро цитоплазма рибосомы митохондрии мембрана",
            "синтез белка транскрипция трансляция дезоксирибонуклеиновая кислота",

            # Exact sciences (Physics, Chemistry, Math)
            "уравнение химической реакции гидролиз нейтрализация окисление восстановление",
            "раствор кислоты основания соли щелочь катализатор температура давление",
            "закон сохранения энергии импульс масса ускорение сила тяжести",
            "кинетическая и потенциальная энергия работа мощность теплота",
            "функция непрерывна на отрезке производная интеграл предел последовательности",
            "теорема коши лагранжа дифференциальное исчисление матрица вектор",
            "классная работа домашняя работа упражнение задача вариант решение ответ",
            "двенадцатое апреля двадцать третье мая первое сентября",

            # Russian grammar rules
            "имя прилагательное образованное от существительного",
            "в кратких прилагательных пишется одна н в полных две нн",
            "соломенный огненный стеклянный оловянный деревянный",
            "причастный оборот деепричастный оборот сложноподчиненное предложение",
            "правописание корней с чередованием гласных приставки пре и при",
        ]

        for sentence in seed_corpus:
            self.train_on_text(sentence)

    def train_on_text(self, text: str) -> None:
        """Add text sequence to n-gram frequency distributions."""
        clean = re.sub(r'[^а-яА-Яa-zA-Z0-9\s\-]', ' ', text.lower())
        words = [w.strip() for w in clean.split() if len(w.strip()) > 0]
        if not words:
            return

        for w in words:
            self.unigrams[w] += 1
            self.total_unigrams += 1
            self.vocab.add(w)
            # Train character trigrams for pseudo-word detection
            padded = f"^^{w}$$"
            for i in range(len(padded) - 2):
                self.char_trigrams[padded[i:i+3]] += 1
                self.total_char_trigrams += 1

        for i in range(len(words) - 1):
            self.bigrams[(words[i], words[i+1])] += 1
            self.total_bigrams += 1

        for i in range(len(words) - 2):
            self.trigrams[(words[i], words[i+1], words[i+2])] += 1
            self.total_trigrams += 1

    def compute_word_log_prob(self, word: str, prev_w1: Optional[str] = None, prev_w2: Optional[str] = None) -> float:
        """
        Compute smoothed interpolated log probability log P(w | prev_w1, prev_w2).
        Uses Jelinek-Mercer / Witten-Bell style linear interpolation.
        """
        word = word.lower().strip()
        vocab_size = max(1000, len(self.vocab))

        # Unigram probability
        c_uni = self.unigrams.get(word, 0)
        p_uni = (c_uni + 0.1) / (self.total_unigrams + 0.1 * vocab_size) if self.total_unigrams > 0 else 1.0 / vocab_size

        # Bigram probability
        p_bi = p_uni
        if prev_w1:
            prev_w1 = prev_w1.lower().strip()
            c_prev1 = self.unigrams.get(prev_w1, 0)
            if c_prev1 > 0:
                c_bi = self.bigrams.get((prev_w1, word), 0)
                p_bi = 0.7 * (c_bi / c_prev1) + 0.3 * p_uni

        # Trigram probability
        p_tri = p_bi
        if prev_w1 and prev_w2:
            prev_w2 = prev_w2.lower().strip()
            c_prev2 = self.bigrams.get((prev_w2, prev_w1), 0)
            if c_prev2 > 0:
                c_tri = self.trigrams.get((prev_w2, prev_w1, word), 0)
                p_tri = 0.6 * (c_tri / c_prev2) + 0.4 * p_bi

        return math.log(max(1e-12, p_tri))

    def compute_char_perplexity_penalty(self, word: str) -> float:
        """
        Detect artificial gibberish / pseudo-words via character trigram perplexity.
        Valid Russian words have high character trigram overlap with natural language.
        """
        w = word.lower().strip()
        if len(w) < 3 or not re.search(r'[а-я]', w):
            return 0.0

        padded = f"^^{w}$$"
        log_prob_sum = 0.0
        n_trigrams = len(padded) - 2

        for i in range(n_trigrams):
            tri = padded[i:i+3]
            count = self.char_trigrams.get(tri, 0)
            p = (count + 0.01) / (self.total_char_trigrams + 0.01 * 5000) if self.total_char_trigrams > 0 else 1e-6
            log_prob_sum += math.log(max(1e-12, p))

        avg_log_prob = log_prob_sum / n_trigrams
        # Severe penalty for extremely unlikely letter combinations
        if avg_log_prob < -9.0:
            return (avg_log_prob + 9.0) * 2.5
        return 0.0

    def score_sequence(self, text: str, prev_context: Optional[str] = None) -> float:
        """
        Compute total Language Model log-likelihood score for a sequence of words.
        Incorporates cross-line previous context to score inter-line transitions and word hyphenations.
        """
        if not text or not text.strip():
            return -10.0

        clean = re.sub(r'[^а-яА-Яa-zA-Z0-9\s\-]', ' ', text.lower())
        words = [w.strip() for w in clean.split() if len(w.strip()) > 0]
        if not words:
            return -5.0

        # Extract prior words from preceding line context if available
        context_words: List[str] = []
        is_hyphenated_continuation = False
        hyphen_stem = ""

        if prev_context and prev_context.strip():
            raw_prev_clean = re.sub(r'[^а-яА-Яa-zA-Z0-9\s\-]', ' ', prev_context.lower())
            p_words = [pw.strip() for pw in raw_prev_clean.split() if len(pw.strip()) > 0]
            if p_words:
                context_words = p_words[-2:]
                # Detect hyphenation wrap
                last_token = prev_context.strip().split()[-1]
                if last_token.endswith("-") or last_token.endswith("."):
                    hyphen_stem = re.sub(r'[^а-яА-Яa-zA-Z0-9]', '', last_token).lower()
                    if hyphen_stem:
                        is_hyphenated_continuation = True

        total_score = 0.0
        for i, w in enumerate(words):
            if i == 0 and context_words:
                prev1 = context_words[-1]
                prev2 = context_words[-2] if len(context_words) > 1 else None
            else:
                prev1 = words[i-1] if i > 0 else None
                prev2 = words[i-2] if i > 1 else None

            word_lp = self.compute_word_log_prob(w, prev1, prev2)
            char_penalty = self.compute_char_perplexity_penalty(w)

            # Dictionary bonus for exact vocabulary match
            vocab_bonus = 1.2 if w in self.vocab else 0.0

            # Special reward for completing hyphenated words from previous line
            if i == 0 and is_hyphenated_continuation:
                full_compound = hyphen_stem + w
                if full_compound in self.vocab or (len(full_compound) >= 5 and self.compute_char_perplexity_penalty(full_compound) == 0.0):
                    vocab_bonus += 3.5

            total_score += word_lp + char_penalty + vocab_bonus

        return total_score

    def rescore_candidates(
        self,
        candidates: List[Tuple[str, float]],
        prev_context: Optional[str] = None,
        alpha: float = 0.35,
        word_bonus: float = 0.20,
    ) -> str:
        """
        Rescore beam search / TTA candidate strings using cross-line language model scores.
        """
        if not candidates:
            return ""

        scored_candidates: List[Tuple[float, str]] = []
        for text, ocr_log_prob in candidates:
            if not text.strip():
                continue
            lm_score = self.score_sequence(text, prev_context=prev_context)
            n_words = len(text.split())
            length_bonus = n_words * word_bonus

            # Combined posterior score
            total = (1.0 - alpha) * ocr_log_prob + alpha * lm_score + length_bonus
            scored_candidates.append((total, text))

        if not scored_candidates:
            return candidates[0][0]

        scored_candidates.sort(reverse=True, key=lambda x: x[0])
        return scored_candidates[0][1]


_global_rescorer: Optional[NGramLanguageModelRescorer] = None


def get_language_model_rescorer() -> NGramLanguageModelRescorer:
    """Singleton getter for the global N-gram LM rescorer."""
    global _global_rescorer
    if _global_rescorer is None:
        _global_rescorer = NGramLanguageModelRescorer()
    return _global_rescorer
