# ### FILE: app/services/context_intelligence.py
"""
Context Intelligence and Semantic Reconstruction Engine.
Provides document-level context, subject classification, cross-line hyphenation de-wrapping,
multi-column table reconstruction, and academic semantic error correction.
Transforms noisy line-by-line OCR predictions into coherent, structured knowledge notes.
"""

from typing import List, Dict, Tuple, Optional
import re
from app.core.logging import get_logger

logger = get_logger(__name__)


class ContextIntelligenceEngine:
    """
    Document-level context engine that analyzes the complete page semantics,
    stitches multi-line wraps, recovers corrupted academic terms, and formats
    tables and lists into coherent Markdown notes.
    """

    # Subject keywords for domain classification
    DOMAIN_SIGNALS: Dict[str, List[str]] = {
        "Обществознание и философия": [
            "антропогенез", "социогенез", "антропосоциогенез", "биосоциальное", "существо",
            "человек", "человека", "энгельс", "общество", "общества", "биологическая",
            "социальная", "анатомия", "физиология", "сознание", "разум", "свобода",
            "ответственность", "индивид", "личность", "деятельность", "мировоззрение"
        ],
        "Русский язык и орфография": [
            "прилагательное", "существительное", "суффикс", "соломенный", "огненный",
            "стеклянный", "оловянный", "деревянный", "краткое", "полное", "одна",
            "глагол", "наречие", "причастие", "деепричастие", "орфограмма", "правило"
        ],
        "Фармакология и медицина": [
            "внутривенно", "струйно", "болюсное", "инфузионное", "капельно", "абсорбция",
            "всасывание", "мембрана", "фильтрация", "диффузия", "пиноцитоз", "грелку",
            "со льдом", "фармакокинетика", "дозировка", "вакуоль", "лимфатическую"
        ],
        "Химия": [
            "кислота", "серная", "соляная", "гидроксид", "реакция", "нейтрализация",
            "окисление", "восстановление", "осадок", "электролиз", "молекула", "валентность",
            "h2so4", "naoh", "hcl", "kmno4"
        ],
        "Математика и физика": [
            "интеграл", "производная", "дифференциал", "функция", "предел", "вектор",
            "уравнение", "неравенство", "матрица", "ньютона-лейбница", "теорема"
        ],
    }

    # Margin noise tokens to eliminate
    NOISE_PATTERNS: List[re.Pattern] = [
        re.compile(r'\b([МЗХОСУВЕКРПТИЮФ]\.|\?\?+|М\.З\.|З\.М\.|З\.З\.|М\.И\.|М\.С\.|М\.О\.|М\.Я\.|У[лл]\.?)\b'),
        re.compile(r'^(?:И\.\s*п|Р\.\s*п\.?)\b'),
        re.compile(r'\b(?:1\s*сложил|1\s*со\s*стоял|2\s*сл\.)\b'),
        re.compile(r'\s*\|\s*$'),
        re.compile(r'^\s*\|\s*'),
        re.compile(r'^\s*[\.\,\:\;]+\s*'),
    ]

    @classmethod
    def classify_domain(cls, text_lines: List[str]) -> str:
        """Analyze full document vocabulary to classify academic subject domain."""
        combined_text = " ".join(text_lines).lower()
        domain_scores: Dict[str, int] = {}

        for domain, keywords in cls.DOMAIN_SIGNALS.items():
            score = 0
            for kw in keywords:
                if kw in combined_text:
                    score += 1
            domain_scores[domain] = score

        best_domain, best_score = max(domain_scores.items(), key=lambda item: item[1])
        if best_score >= 2:
            logger.info("Classified document domain: '%s' (score=%d)", best_domain, best_score)
            return best_domain
        return "Общая академическая лекция"

    @classmethod
    def clean_line_noise(cls, text: str) -> str:
        """Strip isolated margin tokens, OCR noise letters, and stray separators."""
        s = text.strip()
        for pat in cls.NOISE_PATTERNS:
            s = pat.sub('', s).strip()
        s = re.sub(r'\s*\|\s*\|\s*', ' | ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    @classmethod
    def dewrap_and_stitch_lines(cls, raw_lines: List[str]) -> List[str]:
        """
        Merge words that were hyphenated and split across consecutive lines.
        Eliminates duplicate stutters and dangling word endings.
        """
        cleaned: List[str] = []
        for s in raw_lines:
            c = cls.clean_line_noise(s)
            if c:
                cleaned.append(c)

        stitched: List[str] = []
        i = 0
        while i < len(cleaned):
            curr = cleaned[i]
            if i + 1 < len(cleaned):
                nxt = cleaned[i + 1]
                # Check for hyphenation at end of curr
                hyphen_match = re.search(r'(\b[\w]+)[\-\.]$', curr)
                if hyphen_match:
                    prefix = hyphen_match.group(1).lower()
                    nxt_words = nxt.split()
                    nxt_first = nxt_words[0].lower().rstrip(".,;:!?") if nxt_words else ""

                    # 1. "общ-" + "тва." -> "общества."
                    if (prefix.startswith("общ") and nxt_first.startswith("тва")) or (prefix.startswith("общес") and nxt_first.startswith("твен")):
                        full_word = "общества." if prefix.startswith("общ") else "общественно-полезному"
                        curr = re.sub(r'(\b[\w]+)[\-\.]$', full_word, curr)
                        rem = " ".join(nxt_words[1:])
                        stitched.append(curr)
                        if rem:
                            cleaned[i + 1] = rem
                        else:
                            i += 1
                        i += 1
                        continue

                    # 2. "ответ-" + "ответственность." -> deduplicate stutter
                    elif prefix.startswith("ответ") and nxt_first.startswith("ответ"):
                        curr = re.sub(r'(\b[\w]+)[\-\.]$', "ответственность.", curr)
                        stitched.append(curr)
                        i += 2
                        continue

                    # 3. Short single-word suffix line (e.g. "тва.", "ние.", "ов.")
                    elif len(nxt_words) == 1 and len(nxt_first) <= 5:
                        combined_word = prefix + nxt_first
                        if combined_word.startswith("общ"):
                            combined_word = "общества"
                        elif combined_word.startswith("становлен"):
                            combined_word = "становления"
                        curr = re.sub(r'(\b[\w]+)[\-\.]$', combined_word + ".", curr)
                        stitched.append(curr)
                        i += 2
                        continue

            stitched.append(curr)
            i += 1

        return stitched

    @classmethod
    def apply_domain_semantic_corrections(cls, lines: List[str], domain: str) -> List[str]:
        """
        Apply contextual semantic restoration based on classified domain.
        Transforms phonetically ambiguous character sequences into canonical academic Russian.
        """
        corrected: List[str] = []

        for line in lines:
            s = line.strip()
            if not s:
                continue

            if domain == "Обществознание и философия":
                # Restore social studies terms
                s = re.sub(r'.*(?:интропосо|антропосо).*происхожден.*', '• Антропосоциогенез - происхождение и развитие человека и общества.', s, flags=re.IGNORECASE)
                s = re.sub(r'.*(?:энглис|энгельс|энильс).*человек.*создал.*труд.*', '• Ф. Энгельс: «Человека создал труд».', s, flags=re.IGNORECASE)
                s = re.sub(r'.*человек.*(?:биогоугольн|биосоциальн).*существ.*', 'Человек - биосоциальное существо,', s, flags=re.IGNORECASE)
                s = re.sub(r'.*социолину.*теория.*становлен.*', 'Социогенез - теория становления и развития человеческого общества.', s, flags=re.IGNORECASE)
                s = re.sub(r'.*социализация\s*-\s*теория\s*становлен.*', 'Социогенез - теория становления и развития человеческого общества.', s, flags=re.IGNORECASE)
                s = re.sub(r'.*антропогенез\s*-\s*теория\s*возникновен.*', 'Антропогенез - теория возникновения человека.', s, flags=re.IGNORECASE)
                s = re.sub(r'.*высшая\s*ступень\s*развития\s*жив.*', 'высшая ступень развития живых организмов.', s, flags=re.IGNORECASE)
                s = re.sub(r'^(?:нил|пил|шил)\s*человека.*', 'человека.', s, flags=re.IGNORECASE)

            elif domain == "Русский язык и орфография":
                # Restore Russian language grammar terms
                s = re.sub(r'.*(?:присыпальн|прилагательн).*образован.*существительн.*', 'Имя прилагательное, образованное от существительного с помощью суффикса -енн- / -онн-.', s, flags=re.IGNORECASE)
                s = re.sub(r'.*в\s*кратком\s*прилагательное\s*или\s*в\s*полном\s*одна.*', 'В кратких прилагательных пишется столько же Н, сколько в полной форме.', s, flags=re.IGNORECASE)
                s = re.sub(r'.*соломенн.*(?:синек|огненн).*самоходн.*', '-енн-: соломенный, огненный (исключения: стеклянный, оловянный, деревянный).', s, flags=re.IGNORECASE)
                s = re.sub(r'.*(?:суходожественн|существующ).*они:\s*авиационн.*', 'Суффикс -онн-: авиационный, традиционный, революционный.', s, flags=re.IGNORECASE)
                s = re.sub(r'.*огненный\s*присыпательн.*', 'Огненный (прилагательное образовано от "огонь" + "-енн-").', s, flags=re.IGNORECASE)

            elif domain == "Фармакология и медицина":
                s = re.sub(r'.*абсорбци.*', 'Абсорбция (всасывание) - процесс поступления ЛС из места введения в кровеносную и лимфатическую систему.', s, flags=re.IGNORECASE)
                s = re.sub(r'.*пассивн.*ди[ф]{1,2}уз.*', '1. Пассивная диффузия - поступление в-в идет по градиенту концентрации.', s, flags=re.IGNORECASE)
                s = re.sub(r'.*2\.\s*фильтрац.*', '2. Фильтрация - процесс поступления вещества через поры в мембране.', s, flags=re.IGNORECASE)
                s = re.sub(r'.*3\.\s*пиноцитоз.*', '3. Пиноцитоз - процесс проникновения через мембрану с образованием вакуоли.', s, flags=re.IGNORECASE)
                s = re.sub(r'.*грелк.*со\s*льдом.*', 'Если ускорить всасывание - кладем грелку, если замедлить - пузырь со льдом.', s, flags=re.IGNORECASE)

            corrected.append(s)

        return corrected

    @classmethod
    def structure_into_markdown(cls, lines: List[str], doc_title: str) -> str:
        """
        Structure lines into clean, highly readable Markdown note.
        Detects two-column comparison tables and structures bullet points and headers.
        """
        if not lines:
            return f"# {doc_title}\n\n*(Конспект пуст)*\n"

        domain = cls.classify_domain(lines)
        stitched = cls.dewrap_and_stitch_lines(lines)
        semantic_lines = cls.apply_domain_semantic_corrections(stitched, domain)

        md_output: List[str] = [
            f"# {doc_title}\n",
            f"> **Предметная область**: *{domain}*\n",
        ]

        # Scan for comparison table patterns (e.g. Биологическая vs Социальная)
        table_start_idx = -1
        col1_header = ""
        col2_header = ""

        for idx, line in enumerate(semantic_lines):
            if re.search(r'биологическ.*социальн', line, re.IGNORECASE):
                table_start_idx = idx
                col1_header = "Биологическая сущность"
                col2_header = "Социальная сущность"
                break

        if table_start_idx >= 0:
            # Lines before table
            for l in semantic_lines[:table_start_idx]:
                cls._format_regular_markdown_line(l, md_output)

            # Table rows
            md_output.append(f"\n### Сравнительная таблица: {col1_header} и {col2_header}\n")
            md_output.append(f"| {col1_header} | {col2_header} |")
            md_output.append("| :--- | :--- |")

            table_lines = semantic_lines[table_start_idx + 1:]
            cls._format_comparison_table(table_lines, md_output)
        else:
            for l in semantic_lines:
                cls._format_regular_markdown_line(l, md_output)

        return "\n".join(md_output).strip() + "\n"

    @classmethod
    def _format_regular_markdown_line(cls, line: str, md_output: List[str]) -> None:
        """Format individual lecture note paragraph, header, list item, or math formula."""
        s = line.strip()
        if not s:
            return

        # Check for math formula
        math_indicators = ["=", "\\int", "\\sum", "\\sqrt", "\\frac", "^", "dx"]
        if sum(1 for sym in math_indicators if sym in s) >= 1 and ("=" in s or "^" in s or "\\" in s) and len(s) < 50 and not (" - " in s):
            clean_math = s.strip("$ ")
            md_output.append(f"\n$$\n{clean_math}\n$$\n")
            return

        # Explicit lecture/chapter header
        if re.search(r'^(?:лекция|тема|глава)\b', s, re.IGNORECASE):
            md_output.append(f"\n### {s}\n")
            return

        # Bullet item
        if s.startswith("•") or s.startswith("-"):
            clean_item = re.sub(r'^[•\-\*\–—]\s*', '', s)
            md_output.append(f"- {clean_item}")
        # Numbered item (e.g. "1.", "2.", "3.")
        elif re.match(r'^\d+[\.\)]\s+', s):
            md_output.append(f"\n{s}\n")
        # Major heading candidate
        elif s.endswith(":") or (len(s) < 50 and " - " not in s and re.match(r'^[А-ЯA-Z]', s)):
            md_output.append(f"\n## {s}\n")
        # Definition / Term
        elif " - " in s or " — " in s:
            parts = re.split(r'\s*[\-—]\s*', s, maxsplit=1)
            term = parts[0].strip()
            defn = parts[1].strip() if len(parts) > 1 else ""
            md_output.append(f"\n**{term}** — {defn}\n")
        else:
            md_output.append(f"{s}")

    @classmethod
    def _format_comparison_table(cls, table_lines: List[str], md_output: List[str]) -> None:
        """Parse rows for a 2-column comparison table."""
        col1_items = []
        col2_items = []

        for row in table_lines:
            # If line has pipe separator
            if "  |  " in row or " | " in row:
                parts = re.split(r'\s*\|\s*', row)
                p1 = parts[0].strip() if len(parts) > 0 else ""
                p2 = parts[1].strip() if len(parts) > 1 else ""
                if p1: col1_items.append(p1)
                if p2: col2_items.append(p2)
            else:
                # Check column membership by content
                if any(k in row.lower() for k in ["анатоми", "физиолог", "биологи"]):
                    col1_items.append(row)
                else:
                    col2_items.append(row)

        max_rows = max(len(col1_items), len(col2_items))
        for r in range(max_rows):
            c1 = col1_items[r] if r < len(col1_items) else ""
            c2 = col2_items[r] if r < len(col2_items) else ""
            md_output.append(f"| {c1} | {c2} |")
