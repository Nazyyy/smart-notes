# ### FILE: app/services/paragraph_context_corrector.py
"""
Paragraph-Level Context and Coherence Corrector.
Stitches cross-line hyphenated words, binds hanging prepositions,
and unifies fragmented line segments into coherent academic paragraphs.
"""

from typing import List, Tuple, Optional
import re

from app.core.logging import get_logger
from app.ml.handwriting_confusion import get_handwriting_confusion_corrector

logger = get_logger(__name__)


class ParagraphContextCorrector:
    """
    Analyzes sequences of lines in a page/document to assemble coherent paragraphs,
    seamlessly stitching words broken across lines and maintaining grammatical flow.
    """

    # Russian hanging prepositions and conjunctions
    HANGING_CONNECTORS = {
        "в", "во", "с", "со", "к", "ко", "о", "об", "обо", "у", "и", "а", "но",
        "на", "по", "из", "до", "от", "за", "под", "над", "при", "без", "для"
    }

    def __init__(self) -> None:
        self.confusion = get_handwriting_confusion_corrector()

    def stitch_hyphenated_words(self, lines: List[str]) -> List[str]:
        """
        Stitch words broken by hyphens or dashes across consecutive lines.
        E.g. Line 1 ends with 'лимфати-', Line 2 begins with 'ческую' -> 'лимфатическую'.
        """
        if not lines:
            return []

        stitched: List[str] = []
        i = 0
        n = len(lines)

        while i < n:
            curr = lines[i].strip()
            if not curr:
                stitched.append("")
                i += 1
                continue

            # Check if line ends with a hyphen/dash
            hyphen_match = re.search(r'([А-Яа-яЁёA-Za-z]+)\s*[\-\–—=]\s*$', curr)
            if hyphen_match and i + 1 < n:
                next_line = lines[i + 1].strip()
                next_match = re.match(r'^([А-Яа-яЁёA-Za-z]+)(.*)$', next_line)
                if next_match:
                    prefix_word = hyphen_match.group(1)
                    suffix_word = next_match.group(1)
                    remainder_next = next_match.group(2)

                    combined_word = prefix_word + suffix_word
                    # Replace hyphenated end of curr line with the stitched word
                    curr_prefix = curr[:hyphen_match.start(1)]
                    merged_line = f"{curr_prefix}{combined_word}{remainder_next}"
                    stitched.append(merged_line)
                    logger.debug("Stitched hyphenated word: '%s-' + '%s' -> '%s'", prefix_word, suffix_word, combined_word)
                    i += 2
                    continue

            stitched.append(curr)
            i += 1

        return stitched

    def bind_hanging_prepositions(self, lines: List[str]) -> List[str]:
        """
        If a line ends with a dangling preposition ('в', 'на', 'из', etc.) and the next
        line begins with a lowercase word, bind them together smoothly.
        """
        if not lines:
            return []

        bound: List[str] = []
        i = 0
        n = len(lines)

        while i < n:
            curr = lines[i].strip()
            if not curr:
                bound.append("")
                i += 1
                continue

            words = curr.split()
            if words and words[-1].lower() in self.HANGING_CONNECTORS and i + 1 < n:
                next_line = lines[i + 1].strip()
                # Check if next line is a continuation (starts with lowercase or alnum)
                if next_line and (next_line[0].islower() or next_line[0].isdigit()):
                    hanging_prep = words[-1]
                    line_without_prep = " ".join(words[:-1])
                    new_next_line = f"{hanging_prep} {next_line}"
                    if line_without_prep:
                        bound.append(line_without_prep)
                    bound.append(new_next_line)
                    i += 2
                    continue

            bound.append(curr)
            i += 1

        return bound

    def merge_broken_sentences(self, lines: List[str]) -> List[str]:
        """
        Merge lines that clearly belong to the same sentence (no punctuation at end of line 1,
        and line 2 starts with lowercase).
        """
        if not lines:
            return []

        merged: List[str] = []
        buffer = ""

        for line in lines:
            s = line.strip()
            if not s:
                if buffer:
                    merged.append(buffer)
                    buffer = ""
                continue

            # Check if this is a header or list item that should never be merged
            is_header_or_bullet = (
                s.startswith("#") or
                s.startswith("•") or
                s.startswith("- ") or
                bool(re.match(r'^\d+[\.\)]\s+', s)) or
                s.startswith("$$")
            )

            if not buffer:
                buffer = s
            elif is_header_or_bullet:
                merged.append(buffer)
                buffer = s
            elif buffer.endswith(":") or buffer.startswith("$$") or buffer.endswith("$$"):
                merged.append(buffer)
                buffer = s
            elif buffer[-1] in ".!?" and s[0].isupper():
                # Definite sentence boundary
                merged.append(buffer)
                buffer = s
            elif not (buffer[-1] in ".!?:") and s[0].islower():
                # Continuation of current sentence
                buffer = f"{buffer} {s}"
            else:
                merged.append(buffer)
                buffer = s

        if buffer:
            merged.append(buffer)

        return merged

    def restore_capitalization(self, text: str) -> str:
        """
        Ensure sentences starting after '. ', '! ', '? ' begin with a capital letter.
        """
        def _cap_match(m):
            return m.group(1) + m.group(2).upper()

        res = re.sub(r'([\.\!\?]\s+)([а-яa-z])', _cap_match, text)
        if res and res[0].islower():
            res = res[0].upper() + res[1:]
        return res

    def process_document_lines(self, lines: List[str]) -> List[str]:
        """
        Full pipeline:
        1. Stitch hyphens across lines
        2. Bind hanging prepositions
        3. Merge broken sentences
        4. Normalize capitalization and spacing
        """
        step1 = self.stitch_hyphenated_words(lines)
        step2 = self.bind_hanging_prepositions(step1)
        step3 = self.merge_broken_sentences(step2)
        return [self.restore_capitalization(l) for l in step3]


_corrector_instance: Optional[ParagraphContextCorrector] = None


def get_paragraph_context_corrector() -> ParagraphContextCorrector:
    """Singleton getter for ParagraphContextCorrector."""
    global _corrector_instance
    if _corrector_instance is None:
        _corrector_instance = ParagraphContextCorrector()
    return _corrector_instance
