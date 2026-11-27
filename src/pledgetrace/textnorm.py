"""Comparison-only normalisation. Never used to rewrite exported text."""

from __future__ import annotations

import unicodedata


def norm_with_map(text: str) -> tuple[str, list[int]]:
    """Casefolded, accent-free alphanumerics plus the source index of each char."""
    out: list[str] = []
    index: list[int] = []
    for position, char in enumerate(text):
        for piece in unicodedata.normalize("NFKD", char):
            if unicodedata.combining(piece):
                continue
            if piece.isalnum():
                for folded in piece.casefold():
                    out.append(folded)
                    index.append(position)
    return "".join(out), index


def norm(text: str) -> str:
    return norm_with_map(text)[0]


def norm_words(text: str) -> str:
    """Casefolded, accent-free text that keeps word boundaries."""
    plain = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in plain if not unicodedata.combining(char))
