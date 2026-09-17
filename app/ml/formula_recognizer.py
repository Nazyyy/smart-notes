# ### FILE: app/ml/formula_recognizer.py
"""
Academic Formula and Scientific Expression Recognizer (HTR -> LaTeX/KaTeX).
Detects and translates mathematical equations, physical laws, and chemical reactions
into standard LaTeX notation formatted for Markdown KaTeX rendering ($...$ and $$...$$).
"""

from typing import List, Tuple, Optional
import re

from app.core.logging import get_logger

logger = get_logger(__name__)


class AcademicFormulaRecognizer:
    """
    Parses handwritten OCR text into structured LaTeX expressions.
    Supports fractions, powers, roots, limits, integrals, chemical subscripts, and Greek symbols.
    """

    # Common chemical compounds
    COMMON_CHEMICAL_PATTERNS = [
        (r'(?<![a-zA-Z])H2SO4(?![a-zA-Z])', r'\text{H}_2\text{SO}_4'),
        (r'(?<![a-zA-Z])KMnO4(?![a-zA-Z])', r'\text{KMnO}_4'),
        (r'(?<![a-zA-Z])HNO3(?![a-zA-Z])', r'\text{HNO}_3'),
        (r'(?<![a-zA-Z])H3PO4(?![a-zA-Z])', r'\text{H}_3\text{PO}_4'),
        (r'(?<![a-zA-Z])HCl(?![a-zA-Z])', r'\text{HCl}'),
        (r'(?<![a-zA-Z])NaOH(?![a-zA-Z])', r'\text{NaOH}'),
        (r'(?<![a-zA-Z])KOH(?![a-zA-Z])', r'\text{KOH}'),
        (r'(?<![a-zA-Z])NaCl(?![a-zA-Z])', r'\text{NaCl}'),
        (r'(?<![a-zA-Z])H2O(?![a-zA-Z])', r'\text{H}_2\text{O}'),
        (r'(?<![a-zA-Z])CO2(?![a-zA-Z])', r'\text{CO}_2'),
        (r'(?<![a-zA-Z])NH3(?![a-zA-Z])', r'\text{NH}_3'),
        (r'(?<![a-zA-Z])CH4(?![a-zA-Z])', r'\text{CH}_4'),
        (r'(?<![a-zA-Z])O2(?![a-zA-Z])', r'\text{O}_2'),
        (r'(?<![a-zA-Z])H2(?![a-zA-Z])', r'\text{H}_2'),
        (r'(?<![a-zA-Z])N2(?![a-zA-Z])', r'\text{N}_2'),
        (r'(?<![a-zA-Z])Cl2(?![a-zA-Z])', r'\text{Cl}_2'),
        (r'(?<![a-zA-Z])Ca\(OH\)2(?![a-zA-Z])', r'\text{Ca(OH)}_2'),
        (r'(?<![a-zA-Z])BaSO4(?![a-zA-Z])', r'\text{BaSO}_4'),
        (r'(?<![a-zA-Z])AgCl(?![a-zA-Z])', r'\text{AgCl}'),
        (r'(?<![a-zA-Z])CH3-COOH(?![a-zA-Z])', r'\text{CH}_3\text{-COOH}'),
        (r'(?<![a-zA-Z])C6H12O6(?![a-zA-Z])', r'\text{C}_6\text{H}_{12}\text{O}_6'),
    ]

    # Greek letters and physical quantities
    GREEK_AND_PHYSICAL_MAPPINGS = [
        (r'\bDelta\b|\bдельта\b', r'\Delta'),
        (r'\balpha\b|\bальфа\b', r'\alpha'),
        (r'\bbeta\b|\bбета\b', r'\beta'),
        (r'\bgamma\b|\bгамма\b', r'\gamma'),
        (r'\blambda\b|\bлямбда\b', r'\lambda'),
        (r'\bomega\b|\bомега\b', r'\omega'),
        (r'\bOmega\b', r'\Omega'),
        (r'\bphi\b|\bфи\b', r'\varphi'),
        (r'\bpsi\b|\bпси\b', r'\psi'),
        (r'\btheta\b|\bтета\b', r'\theta'),
        (r'\bsigma\b|\bсигма\b', r'\sigma'),
        (r'\bpi\b|\bпи\b', r'\pi'),
        (r'\beps\b|\bepsilon\b|\bэпсилон\b', r'\varepsilon'),
        (r'\bnu\b|\bню\b', r'\nu'),
        (r'\bmu\b|\bмю\b', r'\mu'),
        (r'\bro\b|\brho\b|\bро\b', r'\rho'),
    ]

    # Mathematical symbols and operators
    OPERATOR_MAPPINGS = [
        (r'<=>|<->', r'\Leftrightarrow'),
        (r'->|-->', r'\rightarrow'),
        (r'!=', r'\neq'),
        (r'>=', r'\ge'),
        (r'<=', r'\le'),
        (r'\+-|±', r'\pm'),
        (r'~=|≈', r'\approx'),
        (r'\bint\b|\bинтеграл\b', r'\int'),
        (r'\bsum\b|\bсумма\b', r'\sum'),
        (r'\binfinity\b|\binf\b|\bбесконечность\b', r'\infty'),
    ]

    def is_formula_line(self, text: str) -> bool:
        """
        Check if an entire line represents an academic math or chemistry equation.
        """
        s = text.strip()
        if not s:
            return False

        # If line contains explicit formula markers
        math_indicators = ["=", "+", "-", "*", "/", "^", "_", "->", "<=>", "\\sqrt", "sqrt", "lim", "int", "sin", "cos", "tg", "ln"]
        russian_letters = len(re.findall(r'[а-яА-ЯёЁ]', s))
        total_chars = max(1, len(s.replace(" ", "")))

        # Explicit equation structure (e.g. F = m * a, y = x^2 + 1, H2 + O2 -> H2O)
        has_relational = bool(re.search(r'(=|!=|<=|>=|<|>|->|<=>)', s))
        has_symbols = sum(1 for sym in math_indicators if sym in s)

        # High density of math tokens with low Russian text, or standard formula pattern
        if has_relational and (russian_letters / total_chars < 0.45 or has_symbols >= 2):
            return True

        if s.startswith("$$") or s.endswith("$$"):
            return True

        return False

    def convert_chemistry(self, text: str) -> str:
        """Format chemical formulas with proper subscripts."""
        res = text
        for pat, repl in self.COMMON_CHEMICAL_PATTERNS:
            res = re.sub(pat, lambda m, r=repl: r, res)
        # Generic chemical element subscripting: e.g. Fe2O3, Ca3(PO4)2
        res = re.sub(r'([A-Z][a-z]?)(\d+)', r'\\text{\1}_{\2}', res)
        return res

    def convert_math_to_latex(self, text: str) -> str:
        """
        Converts arithmetic, calculus, and algebraic notation into clean LaTeX.
        """
        s = text.strip()

        # 1. Operators & arrows
        for pat, repl in self.OPERATOR_MAPPINGS:
            s = re.sub(pat, lambda m, r=repl: r, s)

        # 2. Square roots: sqrt(...) -> \sqrt{...}
        s = re.sub(r'\\?sqrt\(([^)]+)\)', r'\\sqrt{\1}', s)
        s = re.sub(r'\\?sqrt([a-zA-Z0-9]+)', r'\\sqrt{\1}', s)

        # 3. Fractions: (a + b) / (c + d) -> \frac{a + b}{c + d}
        s = re.sub(r'\(([^)]+)\)\s*/\s*\(([^)]+)\)', r'\\frac{\1}{\2}', s)
        # Single token fractions: a / b -> \frac{a}{b}
        s = re.sub(r'\b([a-zA-Z0-9_]+)\s*/\s*([a-zA-Z0-9_]+)\b', r'\\frac{\1}{\2}', s)

        # 4. Powers & Superscripts: x^2 -> x^{2}, e^(kx) -> e^{kx}
        s = re.sub(r'\^(\d+)', r'^{\1}', s)
        s = re.sub(r'\^\(([^)]+)\)', r'^{\1}', s)

        # 5. Subscripts: x_0 -> x_{0}, E_k -> E_{k}
        s = re.sub(r'_([a-zA-Z0-9]+)', r'_{\1}', s)

        # 6. Multiplication: * or dot -> \cdot
        s = re.sub(r'\s*\*\s*', r' \\cdot ', s)

        # 7. Greek and physical symbols
        for pat, repl in self.GREEK_AND_PHYSICAL_MAPPINGS:
            s = re.sub(pat, lambda m, r=repl: r, s)

        # 8. Limits: lim x->0 -> \lim_{x \to 0}
        s = re.sub(r'\\?lim\s*(?:при\s*)?([a-zA-Z0-9]+)\s*(?:->|\\rightarrow)\s*([a-zA-Z0-9]+)', r'\\lim_{\1 \\to \2}', s)

        # 9. Clean excessive spaces
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def format_line(self, line: str) -> str:
        """
        Processes a line. If the entire line is a formula, wraps it in $$...$$.
        If the line has inline formula patterns (e.g. 'Формула: y = x^2'), wraps the formula in $...$.
        """
        line_clean = line.strip()
        if not line_clean:
            return ""

        # Check if line already has LaTeX formatting
        if "$$" in line_clean:
            return line_clean

        # 1. Pure formula line
        if self.is_formula_line(line_clean):
            # Check if chemical reaction
            if re.search(r'\b(H2SO4|NaOH|HCl|KMnO4|NaCl|H2O|CO2|O2|H2)\b', line_clean):
                latex_expr = self.convert_chemistry(line_clean)
                latex_expr = re.sub(r'->|-->', r'\\rightarrow', latex_expr)
                latex_expr = re.sub(r'<=>|<->', r'\\Leftrightarrow', latex_expr)
            else:
                latex_expr = self.convert_math_to_latex(line_clean)
            return f"$${latex_expr}$$"

        # 2. Inline formula detection after colon or keywords
        # e.g. "Теорема: a^2 + b^2 = c^2" or "Формула F = m * a"
        inline_match = re.search(r'^(.*?:|\b(?:где|формула|закон)\b)\s*([a-zA-Z0-9\(\)\s\+\-\*\/\^\_\=]+)$', line_clean)
        if inline_match:
            prefix = inline_match.group(1)
            expr = inline_match.group(2).strip()
            if any(sym in expr for sym in ["=", "+", "-", "*", "/", "^", "_"]):
                latex_expr = self.convert_math_to_latex(expr)
                return f"{prefix} ${latex_expr}$"

        # 3. Inline chemical compound mentions in text
        # e.g. "При добавлении H2SO4 раствор синеет"
        words = line_clean.split()
        modified = False
        new_words = []
        for w in words:
            for pat, repl in self.COMMON_CHEMICAL_PATTERNS:
                if re.fullmatch(pat, w.strip(".,;:")):
                    punct = w[-1] if w[-1] in ".,;:" else ""
                    w = f"${repl}${punct}"
                    modified = True
                    break
            new_words.append(w)

        if modified:
            return " ".join(new_words)

        return line_clean

    def format_document_lines(self, lines: List[str]) -> List[str]:
        """Format an entire document's lines, inserting LaTeX math and chemical blocks."""
        formatted: List[str] = []
        for line in lines:
            formatted.append(self.format_line(line))
        return formatted


_formula_recognizer: Optional[AcademicFormulaRecognizer] = None


def get_formula_recognizer() -> AcademicFormulaRecognizer:
    """Singleton getter for AcademicFormulaRecognizer."""
    global _formula_recognizer
    if _formula_recognizer is None:
        _formula_recognizer = AcademicFormulaRecognizer()
    return _formula_recognizer
