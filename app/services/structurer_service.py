# ### FILE: app/services/structurer_service.py
"""
Intelligent Document Structurer and Markdown/LaTeX Formatter.
Analyzes layout geometry, indentation, and text tokens to generate structured notes.
"""

import re
from typing import List, Optional
from app.models.entities import TextLine, ExportFormat
from app.core.logging import get_logger

logger = get_logger(__name__)


class NoteStructurerService:
    """Transforms raw OCR line transcriptions and bounding boxes into structured documents."""

    HEADER_PATTERNS = [
        re.compile(r"^(лекция|тема|глава|определение|теорема|свойство|доказательство|пример)\b", re.IGNORECASE),
        re.compile(r"^(lecture|topic|chapter|definition|theorem|proof|example)\b", re.IGNORECASE),
        re.compile(r"^\d+(\.\d+)*\s+[А-ЯA-Z]"),
    ]

    MATH_INDICATORS = [
        "=", "\\int", "\\sum", "\\sqrt", "\\frac", "\\alpha", "\\beta", "\\gamma",
        "\\pi", "\\sigma", "\\lambda", "^", "_", "\\approx", "\\pm", "\\le", "\\ge",
        "dx", "dy", "dt", "\\lim", "\\partial", "\\infty",
    ]

    BULLET_PATTERNS = [
        re.compile(r"^[\-\*•–—]\s+"),
        re.compile(r"^\d+[\.\)]\s+"),
        re.compile(r"^[a-zA-Zа-яА-Я][\.\)]\s+"),
    ]

    def classify_line(self, text: str, bbox_w: int, bbox_h: int, avg_height: float) -> tuple[str, int]:
        """
        Classify line semantic type:
        Returns (block_type, indent_level)
        block_type in {'h1', 'h2', 'bullet', 'math', 'paragraph'}
        """
        stripped = text.strip()
        if not stripped:
            return "empty", 0

        # Check for math formula
        math_matches = sum(1 for sym in self.MATH_INDICATORS if sym in stripped)
        if math_matches >= 2 or (math_matches >= 1 and ("=" in stripped or "^" in stripped)):
            return "math", 0

        # Check for major headers (larger font height or explicit keyword)
        if bbox_h > avg_height * 1.35 and len(stripped) < 60:
            return "h1", 0

        for pattern in self.HEADER_PATTERNS:
            if pattern.search(stripped):
                return "h2", 0

        if stripped.endswith(":") and len(stripped) < 50:
            return "h2", 0

        # Check for bullet points
        for bullet in self.BULLET_PATTERNS:
            if bullet.match(stripped):
                return "bullet", 0

        return "paragraph", 0

    def calculate_indentation(self, left_x: int, min_x: int, step_px: int = 35) -> int:
        """Calculate logical indentation depth from left margin coordinate."""
        delta = max(0, left_x - min_x)
        return min(delta // step_px, 4)

    def structure_lines_to_markdown(
        self,
        lines: List[TextLine],
        document_title: str = "Конспект лекции",
    ) -> str:
        """
        Compile sorted page lines into a clean, GitHub-flavored Markdown document
        complete with headings, math blocks, and list hierarchies.
        """
        if not lines:
            return f"# {document_title}\n\n*(Документ не содержит распознанного текста)*\n"

        # Calculate layout statistics
        heights = [line.bbox_h for line in lines if line.bbox_h > 0]
        avg_h = float(sum(heights) / len(heights)) if heights else 20.0

        xs = [line.bbox_x for line in lines]
        min_x = min(xs) if xs else 0

        md_output: List[str] = [f"# {document_title}\n"]

        for line in lines:
            text = line.recognized_text.strip()
            if not text:
                continue

            block_type, _ = self.classify_line(text, line.bbox_w, line.bbox_h, avg_h)
            indent = self.calculate_indentation(line.bbox_x, min_x)
            indent_prefix = "  " * indent

            # Normalize OCR artifacts in numbering and prefixes
            text = re.sub(r"^(2002|202)\s*", "2. ", text)
            text = re.sub(r"^М\.З\.\s*", "3. ", text)

            if block_type == "h1":
                clean_title = re.sub(r"^#+\s*", "", text)
                clean_title = re.sub(r"^\d+\s+(?=[А-ЯA-Z])", "", clean_title)
                md_output.append(f"\n## {clean_title}\n")
            elif block_type == "h2":
                clean_title = re.sub(r"^#+\s*", "", text)
                md_output.append(f"\n### {clean_title}\n")
            elif block_type == "math":
                # Ensure LaTeX math formatting
                clean_math = text.strip("$ ")
                md_output.append(f"\n$$\n{clean_math}\n$$\n")
            elif block_type == "bullet":
                clean_item = re.sub(r"^[\-\*•–—\d\.\)]+\s*", "", text)
                clean_item = re.sub(r"^[\.\,\-\–—\s]+", "", clean_item).strip()
                md_output.append(f"{indent_prefix}- {clean_item}")
            else:
                md_output.append(f"{indent_prefix}{text}\n")

        return "\n".join(md_output).strip() + "\n"

    def structure_lines_to_plaintext(
        self,
        lines: List[TextLine],
        document_title: str = "Конспект",
    ) -> str:
        """Compile lines into clean unformatted plain text."""
        lines_text = [f"=== {document_title} ===\n"]
        for line in lines:
            text = line.recognized_text.strip()
            if text:
                lines_text.append(text)
        return "\n".join(lines_text) + "\n"

    def export(
        self,
        lines: List[TextLine],
        export_format: ExportFormat,
        document_title: str = "Конспект",
    ) -> str:
        """Dispatch formatting according to target ExportFormat."""
        if export_format == ExportFormat.MARKDOWN:
            return self.structure_lines_to_markdown(lines, document_title)
        elif export_format == ExportFormat.TXT:
            return self.structure_lines_to_plaintext(lines, document_title)
        elif export_format == ExportFormat.LATEX:
            # Generate minimal LaTeX document
            body = self.structure_lines_to_markdown(lines, document_title)
            latex_header = "\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage{amsmath,amssymb}\n\\begin{document}\n"
            latex_footer = "\n\\end{document}\n"
            return f"{latex_header}{body}{latex_footer}"
        elif export_format == ExportFormat.JSON:
            import json
            items = [
                {
                    "line_index": l.line_index,
                    "bbox": [l.bbox_x, l.bbox_y, l.bbox_w, l.bbox_h],
                    "text": l.recognized_text,
                    "confidence": l.confidence,
                }
                for l in lines
            ]
            return json.dumps({"title": document_title, "lines": items}, ensure_ascii=False, indent=2)

        return self.structure_lines_to_markdown(lines, document_title)
