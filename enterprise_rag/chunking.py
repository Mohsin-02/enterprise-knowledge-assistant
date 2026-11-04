"""A transparent, word-based baseline. Words here are not model tokens."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class TextChunk:
    text: str
    word_start: int
    word_end: int
    char_start: int
    char_end: int

    @property
    def word_count(self) -> int:
        return self.word_end - self.word_start


def split_text(text: str, max_words: int, overlap_words: int) -> list[TextChunk]:
    """Split one page; preserve exact offsets into its cleaned text."""
    if max_words < 1:
        raise ValueError("max_words must be at least 1.")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be >= 0 and less than max_words.")
    words = list(re.finditer(r"\S+", text))
    chunks: list[TextChunk] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        char_start, char_end = words[start].start(), words[end - 1].end()
        chunks.append(TextChunk(text[char_start:char_end], start, end, char_start, char_end))
        if end == len(words):
            break  # Avoid an extra chunk containing only the previous overlap.
        start = end - overlap_words
    return chunks
