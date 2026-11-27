"""Deterministic sub-unit pieces: sentence parts and leading list markers."""

from __future__ import annotations

import re
import string

ABBREVIATIONS = {
    "art",
    "arts",
    "núm",
    "num",
    "nº",
    "sr",
    "sra",
    "sres",
    "dr",
    "dra",
    "etc",
    "pág",
    "págs",
    "pag",
    "pp",
    "ej",
    "vs",
    "aprox",
    "ss",
    "cc",
    "aa",
    "ee",
    "uu",
    "excmo",
    "ilmo",
    "cía",
    "av",
    "avda",
    "tel",
    "apdo",
    "cap",
    "vol",
    "ed",
    "op",
    "cit",
    "cf",
    "fig",
    "km",
    "ptas",
    "pts",
    "mill",
    "sta",
    "sto",
}
BREAK = re.compile(r"[.!?…]+[\"”»’')\]]*\s+(?=[A-ZÁÉÍÓÚÑÀÈÒÏÜÇ¿¡\"“«(\[])")
# \uf000-\uf0ff: Symbol/Wingdings bullets that PDFs map to the private use area
# '!"': a symbol-font bullet that some PDFs map to these two characters
BULLET = re.compile(r"^\s*(?:[•●▪■□◦○·‣⁃►▶→➔➢➤✓✔❖\-–—*»>§\uf000-\uf0ffᙨ\xad‐]+|!\")\s*")
# 3.1 or 12.10 is numbering; 200.000 is a quantity (thousands separator)
NUMBER = re.compile(r"^\s*\(?(\d{1,3}(?:\.\d{1,2})*)\s?[.)ºª\-–]?\s+")
LABELS = list(string.ascii_lowercase)


def sentences(text: str, minimum: int = 30) -> list[tuple[int, int]]:
    """Character spans of sentence parts; one span when the unit is not split."""
    cuts = [0]
    for match in BREAK.finditer(text):
        word = re.search(r"(\w+)$", text[: match.start()])
        if text[match.start()] == "." and word:
            token = word.group(1)
            if (
                token.lower() in ABBREVIATIONS
                or len(token) == 1
                or (token.isdigit() and len(token) <= 3)
            ):
                continue
        if match.end() - cuts[-1] >= minimum and len(text) - match.end() >= minimum:
            cuts.append(match.end())
    cuts = cuts[: len(LABELS)]
    return [(start, end) for start, end in zip(cuts, cuts[1:] + [len(text)])]


def strip_marker(text: str) -> tuple[int, str]:
    """(offset after a leading bullet or number marker, printed number or '')."""
    bullet = BULLET.match(text)
    offset = bullet.end() if bullet else 0
    number = NUMBER.match(text[offset:])
    if number:
        return offset + number.end(), number.group(1)
    return offset, ""
