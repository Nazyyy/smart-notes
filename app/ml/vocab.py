# ### FILE: app/ml/vocab.py
"""
Character Vocabulary and Tokenizer for Multilingual Handwritten Text Recognition.
Supports Cyrillic, Latin, Digits, Punctuation, and Mathematical Symbols.
"""

from typing import List, Dict


class Vocabulary:
    """
    Bidirectional mapping between characters and CTC token indices.
    Token index 0 is reserved for CTC Blank (<BLANK>).
    """

    BLANK_TOKEN = "<BLANK>"
    UNK_TOKEN = "<UNK>"

    # Comprehensive character sets
    CYRILLIC_LOWER = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    CYRILLIC_UPPER = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    LATIN_LOWER = "abcdefghijklmnopqrstuvwxyz"
    LATIN_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    DIGITS = "0123456789"
    PUNCTUATION_MATH = " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

    def __init__(self) -> None:
        self.chars: List[str] = [self.BLANK_TOKEN]

        all_chars = (
            self.CYRILLIC_LOWER
            + self.CYRILLIC_UPPER
            + self.LATIN_LOWER
            + self.LATIN_UPPER
            + self.DIGITS
            + self.PUNCTUATION_MATH
        )

        # De-duplicate while preserving deterministic order
        for ch in all_chars:
            if ch not in self.chars:
                self.chars.append(ch)

        self.chars.append(self.UNK_TOKEN)

        self.char_to_id: Dict[str, int] = {ch: idx for idx, ch in enumerate(self.chars)}
        self.id_to_char: Dict[int, str] = {idx: ch for idx, ch in enumerate(self.chars)}
        self.blank_id: int = 0
        self.unk_id: int = self.char_to_id[self.UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> List[int]:
        """Convert a string into a sequence of integer token indices."""
        return [self.char_to_id.get(ch, self.unk_id) for ch in text]

    def decode(self, indices: List[int]) -> str:
        """Convert a sequence of token indices into text, ignoring blank and unk tokens."""
        result: List[str] = []
        for idx in indices:
            if idx == self.blank_id:
                continue
            char = self.id_to_char.get(idx, "")
            if char not in (self.BLANK_TOKEN, self.UNK_TOKEN):
                result.append(char)
        return "".join(result)

    def get_char(self, idx: int) -> str:
        """Get character at index, or empty string if out of bounds."""
        return self.id_to_char.get(idx, "")


# Global singleton vocabulary instance
VOCAB = Vocabulary()
