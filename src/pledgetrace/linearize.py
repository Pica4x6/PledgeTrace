"""PDF -> text units in reading order, with page, style and position. No inference.

A unit is a paragraph, list item or heading candidate. Units are the only thing a
model is later asked to label, so exported text is always rebuilt from the PDF.
Heuristics here are deliberately generic; `evaluate.ceiling` measures, per
document, how well they agree with a manual reference.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf

from .textnorm import norm

VERSION = "linearize-40"
MARKER = re.compile(
    r"^\s*(?:[•●▪■□◦○·‣⁃►▶→➔➢➤✓✔❖\-–—*»>§\uf000-\uf0ff\xad‐]+"
    r"|\(?\d{1,3}(?:\.\d{1,2})*\s?[.)ºª\-–]"
    r"|\d{1,3}(?:\.\d{1,2})+"  # not 200.000, a quantity
    r"|\d{1,3}(?=\s+[A-ZÁÉÍÓÚÑ¿¡])"
    r"|\(?[a-zA-Z][.)]"
    r"|[IVXLC]{1,6}[.)\-])\s+"
    # a bullet glyph set tight against the first word: «\uf0b7Revalorizar…»
    r"|^\s*[•●▪■◦►▶➢➤✓✔❖\uf000-\uf0ffᙨ](?=[^\W\d_])"
)
TOC = re.compile(r"(?:\.\s?){5,}\s*\d*\s*$|…{3,}\s*\d*\s*$")  # dot leaders of an index
# \uf000-\uf0ff: Symbol/Wingdings bullets mapped to the private use area
BULLET_START = re.compile(
    r"^\s*(?:[•●▪■□◦○·‣⁃►▶→➔➢➤✓✔❖\-–—*»>§\uf000-\uf0ff\xad‐]+\s"
    r"|[•●▪■◦►▶➢➤✓✔❖\uf000-\uf0ffᙨ](?=[^\W\d_]))"
)
END = re.compile(r"[.!?:;…][\"”»’')\]]*\s*$")
BOLD = re.compile(r"bold|black|heavy|semibold|demi|fett", re.I)
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\x85\u2028\u2029]")
LIGATURES = str.maketrans(
    {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
)
SYMBOL_FONT = re.compile(r"wingding|symbol|dingbat|zapf|webding", re.I)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hidden(page: pymupdf.Page) -> set[tuple]:
    """Characters (code, rounded origin) of text painted over by an opaque fill
    drawn after it, e.g. an older version of the page left under a white box.
    Readers do not see it."""
    try:
        area = abs(page.rect)
        fills = [  # opaque boxes smaller than the page (not backgrounds)
            (drawing["seqno"], pymupdf.Rect(drawing["rect"]))
            for drawing in page.get_drawings()
            if drawing.get("fill") is not None
            and (drawing.get("fill_opacity") or 1.0) > 0.99
            and abs(pymupdf.Rect(drawing["rect"])) < 0.9 * area
        ]
        if not fills:
            return set()
        hidden, visible = set(), set()
        for span in page.get_texttrace():
            for char in span["chars"]:
                key = (char[0], round(char[2][0]), round(char[2][1]))
                box = pymupdf.Rect(char[3])
                if any(
                    seqno > span["seqno"] and rect.contains(box)
                    for seqno, rect in fills
                ):
                    hidden.add(key)
                else:
                    visible.add(key)
        return hidden - visible  # a character drawn again on top stays visible
    except (RuntimeError, ValueError, KeyError, IndexError):
        return set()


def _visible_spans(page: pymupdf.Page, flags: int, hidden: set) -> list[dict]:
    """Text blocks with hidden characters removed (rawdict, rebuilt as dict)."""
    blocks = page.get_text("rawdict", sort=False, flags=flags)["blocks"]
    for block in blocks:
        for line in block.get("lines", []):
            spans = []
            for span in line["spans"]:
                chars = span["chars"]
                covered = sum(
                    (ord(c["c"]), round(c["origin"][0]), round(c["origin"][1]))
                    in hidden
                    for c in chars
                )
                if not chars or covered * 2 > len(chars):
                    continue  # a span of the hidden layer
                span = dict(span, text="".join(char["c"] for char in chars))
                spans.append(span)
            line["spans"] = spans
            if spans:
                box = pymupdf.Rect(spans[0]["bbox"])
                for span in spans[1:]:
                    box |= pymupdf.Rect(span["bbox"])
                line["bbox"] = tuple(box)
    return blocks


def _page_lines(page: pymupdf.Page, number: int) -> list[dict]:
    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    hidden = _hidden(page)
    if hidden:
        blocks = _visible_spans(page, flags, hidden)
    else:
        blocks = page.get_text("dict", sort=False, flags=flags)["blocks"]
    lines = []
    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            if not line["spans"]:
                continue
            text = "".join(span["text"] for span in line["spans"])
            text = text.translate(LIGATURES)
            controls = len(CONTROL.findall(text))
            text = CONTROL.sub(" ", text)  # tabs, bells… drawn as separators
            if not text.strip():
                continue
            direction = line.get("dir", (1.0, 0.0))
            sizes: Counter = Counter()
            bold = 0
            lead_bold = None
            symbol = any(
                SYMBOL_FONT.search(span["font"])
                for span in line["spans"]
                if span["text"].strip()
            )
            for span in line["spans"]:
                weight = len(span["text"].strip())
                sizes[round(span["size"] * 2) / 2] += weight
                is_bold = bool(span["flags"] & 16 or BOLD.search(span["font"]))
                if is_bold:
                    bold += weight
                if lead_bold is None and span["text"].strip():
                    lead_bold = is_bold
            total = sum(sizes.values()) or 1
            x0, y0, x1, y1 = line["bbox"]
            lines.append(
                {
                    "page": number,
                    "text": text.strip(),
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "size": sizes.most_common(1)[0][0] if sizes else 0.0,
                    "bold": bold * 2 > total,
                    "rotated": abs(direction[0] - 1.0) > 0.05,
                    "bullet": False,
                    "lead_bold": bool(lead_bold),
                    "symbol": symbol,
                    "controls": controls,
                }
            )
    return lines


def _drop_overprints(lines: list[dict]) -> list[dict]:
    """Some PDFs draw text twice in the same place (fake bold, a justification
    fix, or a hidden copy left under the visible one): drop a fragment lying on
    another line of the same row whose text already contains it."""
    removed = set()
    for index, line in enumerate(lines):
        text = norm(line["text"])
        if not text:
            continue
        for other in lines[:index] + lines[index + 1 :]:
            if id(other) in removed:
                continue
            overlap = min(other["y1"], line["y1"]) - max(other["y0"], line["y0"])
            if overlap <= 0.5 * (line["y1"] - line["y0"]):
                continue
            same = (
                other["text"] == line["text"]
                and lines.index(other) < index
                and abs(other["x0"] - line["x0"]) <= 0.3 * line["size"]
            ) or (  # «1» drawn beside «1.»: the same list number twice
                re.fullmatch(r"\d{1,3}", line["text"])
                and other["text"].rstrip(".)") == line["text"]
                and other["text"] != line["text"]
                and abs(other["x0"] - line["x0"])
                <= 1.0 * max(line["size"], other["size"])
            )
            inside = (
                len(other["text"]) > len(line["text"])
                and other["x0"] - 1 <= line["x0"]
                and line["x1"] <= other["x1"] + 1
                and text in norm(other["text"])
            )
            if same or inside:
                removed.add(id(line))
                break
    return [line for line in lines if id(line) not in removed]


def _join_words(lines: list[dict]) -> list[dict]:
    """Justified lines are sometimes stored one word at a time; gutter detection
    would then see columns between the words. Join fragments of one row that are
    separated by no more than a word space."""

    def list_number(line: dict) -> bool:
        return bool(re.fullmatch(r"\d{1,3}[.)]?", line["text"]))

    def same_row(a: dict, b: dict) -> bool:
        overlap = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
        lower = min(a["y1"] - a["y0"], b["y1"] - b["y0"])
        if a["rotated"] or b["rotated"]:
            return False
        if list_number(a) != list_number(b):  # a list number may be set larger,
            big, small = (a, b) if a["size"] >= b["size"] else (b, a)
            if big["size"] < 1.8 * small["size"]:  # (a much larger one is left
                return overlap > 0.5 * lower  # to _attach_numerals)
            return False
        return abs(a["size"] - b["size"]) <= 0.6 and overlap > 0.7 * lower

    bands: list[list[dict]] = []  # lines of one visual row, whatever their order
    for line in sorted(lines, key=lambda line: (line["y0"], line["x0"])):
        for band in bands:
            if same_row(band[0], line):
                band.append(line)
                break
        else:
            bands.append([line])
    rows: list[list[dict]] = []
    for band in bands:
        band.sort(key=lambda line: line["x0"])
        rows.append([band[0]])
        for line in band[1:]:
            last = rows[-1][-1]
            gap = line["x0"] - last["x1"]
            single_words = all(  # words, not numbers or table cells
                len(part["text"].split()) <= 2 and re.search(r"[^\W\d_]", part["text"])
                for part in (line, last)
            )
            # a word space; wider in a justified line stored one word at a time.
            # Two long pieces are never joined: that is a narrow column gutter.
            short = min(len(line["text"].split()), len(last["text"].split())) <= 3
            # a list number and the text it opens, set apart by an indent: «3   Implantaremos…»
            number = (
                len(rows[-1]) == 1
                and list_number(last)
                and re.match(r"[^\W\d_]", line["text"])
                and (last["text"][-1] in ".)" or line["text"][:1].isupper())
            )
            limit = 3.0 if number else 2.0 if single_words else 1.1
            # pieces of different size share a row only as «number + its text»;
            # a large number after other text belongs to the next column
            matched = number or abs(line["size"] - last["size"]) <= 0.6
            size = min(line["size"], last["size"]) if not number else line["size"]
            if matched and (short or number) and 0 <= gap <= limit * size:
                rows[-1].append(line)
            else:
                rows.append([line])
    joined = []
    for row in rows:
        if len(row) == 1:
            joined.append(row[0])
            continue
        first = dict(row[0])
        first["text"] = " ".join(part["text"] for part in row)
        first["x1"] = row[-1]["x1"]
        first["y0"] = min(part["y0"] for part in row)
        first["y1"] = max(part["y1"] for part in row)
        first["bold"] = sum(len(p["text"]) for p in row if p["bold"]) * 2 > len(
            first["text"]
        )
        first["bullet"] = row[0]["bullet"]
        if list_number(row[0]):  # columns are found from the text, not the number
            first["text_x0"] = row[1]["x0"]
        main = max(row, key=lambda part: len(part["text"]))  # style of the text,
        first["size"] = main["size"]  # not of a larger list number before it
        first["lead_bold"] = main.get("lead_bold", first.get("lead_bold"))
        joined.append(first)
    return joined


def _attach_numerals(lines: list[dict]) -> list[dict]:
    """A large stand-alone number (e.g. «12» at 39 pt) beside or just above a body
    line is the printed number of an entry: keep it at the start of that line.
    Runs after reading order, so the numbers still help to find the columns.
    A large number with no line beside it (chapter or decorative) is dropped."""
    removed: set[int] = set()
    used: set[int] = set()
    sizes = sorted(line["size"] for line in lines) or [0.0]
    usual = sizes[len(sizes) // 2]

    # A column that holds nothing but numbers is the margin of the next column.
    text_cols = {line.get("col") for line in lines if len(line["text"]) > 3}
    only_numbers = {line.get("col") for line in lines} - text_cols

    def opens_entry(line: dict) -> bool:
        """No text right above it in its column, or the text above ends a sentence."""
        above = [
            other
            for other in lines
            if other is not line
            and other.get("col") == line.get("col")
            and other["y1"] <= line["y0"] + 1
            and other["x0"] < line["x1"]
            and other["x1"] > line["x0"]
            and len(other["text"]) > 3
        ]
        if not above:
            return True
        nearest = max(above, key=lambda other: other["y1"])
        gap = line["y0"] - nearest["y1"]
        return gap > 0.5 * line["size"] or bool(END.search(nearest["text"]))

    for numeral in lines:
        text = numeral["text"]
        if not re.fullmatch(r"\d{1,3}\.?", text) or numeral["rotated"]:
            continue
        candidates = [
            line
            for line in lines
            if line is not numeral
            and id(line) not in used
            and (
                line.get("col") == numeral.get("col")
                or numeral.get("col") in only_numbers
                or line["x0"] >= numeral["x1"] - 2  # right beside it, next column
            )
            and not line["text"][:1].islower()  # an entry does not start mid-sentence
            and not line["rotated"]
            and len(line["text"]) > 3
            and numeral["size"] >= 1.8 * line["size"]
            and numeral["x0"] - 2 <= line["x0"] <= numeral["x1"] + 60
            and numeral["y0"] - line["size"] <= line["y0"] <= numeral["y1"]
            and opens_entry(line)
        ]
        target = min(
            candidates, key=lambda line: abs(line["y0"] - numeral["y0"]), default=None
        )
        if target is not None:
            if not target["text"].startswith(text.rstrip(".")):
                target["text"] = f"{text.rstrip('.')}. {target['text']}"
            target["bullet"] = True
            used.add(id(target))
            removed.add(id(numeral))
        elif numeral["size"] >= 1.8 * usual:
            removed.add(id(numeral))  # a large chapter or decorative number
    return [line for line in lines if id(line) not in removed]


HYPHENS = {"-", "‐", "\u00ad", "–"}


def _attach_glyphs(lines: list[dict]) -> list[dict]:
    """Drop stand-alone bullet glyphs (often symbol-font letters such as "q") and
    flag the text line beside them as the start of a list item."""
    removed = set()

    def is_word(line: dict) -> bool:
        """A short word or list number in a text font, e.g. «de», «la», «y», «1)»:
        justified lines are sometimes stored one word at a time. «o» at the start
        of a line is the usual exception: a bullet typed as a letter."""
        text = line["text"]
        if line.get("symbol"):
            return False
        if re.search(r"\d", text):  # «6%», «1.», «12»
            # unless it is the text-font copy of a symbol-font bullet drawn on it
            return not any(
                other.get("symbol")
                and abs(other["x0"] - line["x0"]) < line["size"]
                and min(other["y1"], line["y1"]) - max(other["y0"], line["y0"])
                > 0.5 * (line["y1"] - line["y0"])
                for other in lines
            )
        if not re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ]{1,2}", text):
            return False
        if text.lower() != "o":
            return True
        return any(  # «o» (Spanish "or") inside a line, after another word
            other is not line
            and 0 <= line["x0"] - other["x1"] <= 2 * line["size"]
            and min(other["y1"], line["y1"]) - max(other["y0"], line["y0"])
            > 0.5 * (line["y1"] - line["y0"])
            for other in lines
        )

    glyphs = [
        line
        for line in lines
        if len(line["text"]) <= 2 and not line["rotated"] and not is_word(line)
    ]
    if not glyphs:
        return lines
    for glyph in glyphs:
        height = glyph["y1"] - glyph["y0"]
        if glyph["text"] in HYPHENS:
            # a word-break hyphen typeset apart, at the right end of its line
            owner = next(
                (
                    line
                    for line in lines
                    if line is not glyph
                    and len(line["text"]) > 2
                    and -6 <= glyph["x0"] - line["x1"] <= 6
                    and glyph["x0"] > line["x0"]
                    and min(glyph["y1"], line["y1"]) - max(glyph["y0"], line["y0"])
                    > 0.5 * min(height, line["y1"] - line["y0"])
                ),
                None,
            )
            if owner is not None:
                owner["text"] = owner["text"].rstrip() + "-"
                owner["x1"] = max(owner["x1"], glyph["x1"])
                removed.add(id(glyph))
                continue
        best = None
        for line in lines:
            if line is glyph or len(line["text"]) <= 2 or line["rotated"]:
                continue
            distance = line["x0"] - glyph["x1"]
            overlap = min(glyph["y1"], line["y1"]) - max(glyph["y0"], line["y0"])
            if (  # the same marker drawn twice on top of the line's own marker
                -2 <= glyph["x0"] - line["x0"] <= 3 * line["size"]
                and overlap > 0.5 * min(height, line["y1"] - line["y0"])
                and line["text"].startswith(glyph["text"])
            ):
                best = (-1.0, line)
                break
            if -1 <= distance < 60 and overlap > 0.5 * min(
                height, line["y1"] - line["y0"]
            ):
                if best is None or distance < best[0]:
                    best = (distance, line)
        if best:
            best[1]["bullet"] = True
            removed.add(id(glyph))
    return [line for line in lines if id(line) not in removed]


def _text_x0(line: dict) -> float:
    """Left edge of the text: a list number joined to it may stand in the gutter."""
    return line.get("text_x0", line["x0"])


def _gutters(lines: list[dict], width: float) -> list[float]:
    """X positions of column gutters: empty vertical bands with text on both sides."""
    if len(lines) < 12:
        return []
    left = min(_text_x0(line) for line in lines)
    right = max(line["x1"] for line in lines)
    extent = right - left
    # Full-width lines (titles, banners) cross every gutter by nature: they split
    # the page into bands later and must not hide the gutter here.
    lines = [line for line in lines if line["x1"] - _text_x0(line) <= 0.6 * extent]
    if len(lines) < 12:
        return []
    size = int(width) + 2
    cover = [0] * size
    for line in lines:
        for x in range(max(0, int(_text_x0(line))), min(size, int(line["x1"]) + 1)):
            cover[x] += 1
    tolerance = max(1, int(len(lines) * 0.06))
    found = []
    x = int(left + extent * 0.06)
    stop = int(right - extent * 0.06)
    while x < stop:
        if cover[x] <= tolerance:
            start = x
            while x < stop and cover[x] <= tolerance:
                x += 1
            if x - start >= 6:
                middle = (start + x) / 2
                before = sum(1 for line in lines if line["x1"] <= start + 2)
                after = sum(1 for line in lines if _text_x0(line) >= x - 2)
                if before >= 5 and after >= 5:
                    found.append((x - start, middle))
        x += 1
    found.sort(reverse=True)
    return sorted(middle for _, middle in found[:5])


def _order(lines: list[dict], width: float) -> list[dict]:
    """Reading order: full-width lines split the page into bands; columns inside."""
    gutters = _gutters(lines, width)
    for line in lines:
        line["col"] = 0
    if not gutters:
        return sorted(lines, key=lambda item: (round(item["y0"] / 3), item["x0"]))
    crossers, inside = [], []
    for line in lines:
        if any(_text_x0(line) < g - 3 and line["x1"] > g + 3 for g in gutters):
            crossers.append(line)
        else:
            centre = (_text_x0(line) + line["x1"]) / 2
            line["col"] = 1 + sum(1 for g in gutters if centre > g)
            inside.append(line)
    crossers.sort(key=lambda item: item["y0"])
    ordered: list[dict] = []
    bounds = [float("-inf")] + [c["y0"] for c in crossers] + [float("inf")]
    for index in range(len(bounds) - 1):
        band = [
            line for line in inside if bounds[index] <= line["y0"] < bounds[index + 1]
        ]
        band.sort(key=lambda item: (item["col"], round(item["y0"] / 3), item["x0"]))
        ordered.extend(band)
        if index < len(crossers):
            ordered.append(crossers[index])
    return ordered


def _merge_rows(lines: list[dict]) -> list[dict]:
    """Join fragments on one baseline of one column (e.g. a marker and its text)."""
    merged: list[dict] = []
    for line in lines:
        prev = merged[-1] if merged else None
        if (
            prev
            and prev["page"] == line["page"]
            and prev["col"] == line["col"]
            and line["x0"] >= prev["x1"] - 1
            and (  # a large title beside body text is not one line with it
                len(prev["text"]) <= 3
                or len(line["text"]) <= 3
                or max(prev["size"], line["size"])
                <= 1.3 * min(prev["size"], line["size"])
            )
            and min(prev["y1"], line["y1"]) - max(prev["y0"], line["y0"])
            > 0.5 * min(prev["y1"] - prev["y0"], line["y1"] - line["y0"])
        ):
            if len(line["text"]) > len(prev["text"]):
                prev["size"], prev["bold"] = line["size"], line["bold"]
            prev["text"] += " " + line["text"]
            prev["bullet"] = prev["bullet"] or line["bullet"]
            prev["x1"] = line["x1"]
            prev["y0"], prev["y1"] = (
                min(prev["y0"], line["y0"]),
                max(prev["y1"], line["y1"]),
            )
        else:
            merged.append(line)
    return merged


def _key(line: dict) -> str:
    return re.sub(r"\d+", "0", norm(line["text"]))


def _place(line: dict) -> tuple:
    return (_key(line), round(line["y0"] / 4), round(line["x0"] / 8))


def _in_margin(line: dict, height: float) -> bool:
    return line["y1"] < height * 0.1 or line["y0"] > height * 0.9


def _lead(line: dict) -> tuple:
    """Start of a margin line and its height on the page: a running header whose
    end changes with the chapter, e.g. «Programa 2019 / Salud»."""
    words = line["text"].split()[:2]
    return (re.sub(r"\d+", "0", norm(" ".join(words))), round(line["y0"] / 4))


def _isolated(line: dict, lines: list[dict]) -> bool:
    """No other line right above or below it: not the first line of a paragraph."""
    return not any(
        other is not line
        and other["x0"] < line["x1"]
        and other["x1"] > line["x0"]
        and (
            0 <= other["y0"] - line["y1"] < line["size"]
            or 0 <= line["y0"] - other["y1"] < line["size"]
        )
        for other in lines
    )


def _word_in_text(line: dict, lines: list[dict], usual: float) -> bool:
    """A short word that happens to fall in the same place on several pages
    («y», «de», «los» at the end of justified lines) inside running text: not
    furniture, unlike a short logo or label standing alone («Con»)."""
    key = _key(line)
    return (
        len(key) < 4
        and "0" not in key
        and abs(line["size"] - usual) <= 1.0  # body text, not a logo or slogan
        and any(  # another word of the same line right before or after it
            other is not line
            and abs(other["size"] - line["size"]) <= 0.6
            and min(other["y1"], line["y1"]) - max(other["y0"], line["y0"])
            > 0.7 * (line["y1"] - line["y0"])
            and (
                0 <= line["x0"] - other["x1"] <= 1.5 * line["size"]
                or 0 <= other["x0"] - line["x1"] <= 1.5 * line["size"]
            )
            for other in lines
        )
    )


def _entry_number(line: dict, lines: list[dict]) -> bool:
    """A list number («44.», «7)») with its text right beside it: it can fall in
    the same place on several pages without being a page number."""
    if not re.fullmatch(r"\d{1,3}[.)]?", line["text"]):
        return False

    def numbered(text: dict) -> bool:
        return any(
            re.fullmatch(r"\d{1,3}[.)]?", other["text"])
            and 0 <= text["x0"] - other["x1"] <= 3 * other["size"]
            and min(other["y1"], text["y1"]) - max(other["y0"], text["y0"])
            > 0.5 * (other["y1"] - other["y0"])
            for other in lines
        )

    def line_above(text: dict) -> dict | None:
        height = text["y1"] - text["y0"]
        above = [
            other
            for other in lines
            if other is not text
            and other["y1"] <= text["y0"] + 0.3 * height
            and text["y0"] - other["y1"] < 0.8 * height
            and abs(other["x0"] - text["x0"]) <= 0.5 * text["size"]  # same paragraph,
            and min(other["x1"], text["x1"]) - max(other["x0"], text["x0"])  # not a
            > 0.5 * (text["x1"] - text["x0"])  # heading set further out
        ]
        return max(above, key=lambda other: other["y1"]) if above else None

    def first_line(text: dict) -> bool:
        """The line beside the number opens its entry: nothing just above it, the
        line above ends a sentence, or the lines above lead up to the previous
        numbered entry. A number set beside a later line of a paragraph (centred
        on the entry) would otherwise end up inside the sentence."""
        current = line_above(text)
        if current is None or re.search(r"[.:;!?»”)]\s*$", current["text"]):
            return True
        for _ in range(80):  # walk up the paragraph
            if numbered(current):
                return True
            current = line_above(current)
            if current is None or re.search(r"[.:;!?»”)]\s*$", current["text"]):
                return False  # reached its start: the number belongs up there
        return False

    return any(
        other is not line
        and len(other["text"]) > 3
        and line["size"] >= 0.8 * other["size"]  # not a small page number in the margin
        and 0 <= other["x0"] - line["x1"] <= 3 * line["size"]
        and min(other["y1"], line["y1"]) - max(other["y0"], line["y0"])
        > 0.5 * (line["y1"] - line["y0"])
        and first_line(other)
        for other in lines
    )


def _title_start(line: dict, lines: list[dict], usual: float, repeated) -> bool:
    """A large line that a title of the same size continues right below, e.g.
    «ALGUNAS DE LAS ACCIONES MÁS DESTACADAS» / «DE VOX SOBRE ESPAÑA RURAL»: part
    of a heading even when it repeats on several pages, not a running header.
    The continuation must itself change from page to page (`repeated` is False)."""
    if (
        line["rotated"]
        or line["size"] < 1.3 * usual
        or not re.search(r"[^\W\d]", line["text"])
    ):
        return False
    return any(
        other is not line
        and abs(other["size"] - line["size"]) <= 1.0
        and other["y0"] > line["y0"] + 0.5 * line["size"]
        and other["y0"] - line["y1"] < 0.8 * line["size"]
        and other["x0"] < line["x1"]
        and other["x1"] > line["x0"]
        and _key(other) != _key(line)
        and not repeated(other)
        for other in lines
    )


def _furniture(pages: list[tuple[float, list[dict]]]) -> tuple[set, set, set]:
    """Running headers/footers: repeated margin text, or text repeated in place."""
    margin: Counter = Counter()
    placed: Counter = Counter()
    leads: Counter = Counter()
    for height, lines in pages:
        margin.update({_key(line) for line in lines if _in_margin(line, height)})
        placed.update({_place(line) for line in lines if len(line["text"]) <= 80})
        leads.update(
            {
                _lead(line)
                for line in lines
                if _in_margin(line, height)
                and len(_lead(line)[0]) >= 6
                and len(line["text"]) <= 60
                and _isolated(line, lines)
                and not MARKER.match(line["text"])
            }
        )
    common = {lead for lead, count in leads.items() if count >= 3}
    often = {key for key, count in margin.items() if count >= 3 and key}
    often |= {
        _key(line)
        for height, lines in pages
        for line in lines
        if _in_margin(line, height)
        and len(line["text"]) <= 60
        and _lead(line) in common
        and _isolated(line, lines)
        and not MARKER.match(line["text"])
    }
    twice = {key for key, count in margin.items() if count >= 2 and "0" in key}
    fixed = {key for key, count in placed.items() if count >= 3 and key[0]}
    return often, twice, fixed


def linearize(path: Path) -> dict:
    """Return {meta, units}. Units are numbered from 1 in reading order."""
    path = Path(path).expanduser().resolve()
    pages: list[tuple[float, list[dict]]] = []
    widths = []
    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise ValueError(f"Encrypted PDF is not supported: {path}")
        for index in range(len(document)):
            page = document[index]
            pages.append((page.rect.height, _page_lines(page, index + 1)))
            widths.append(page.rect.width)
    raw_chars = sum(len(line["text"]) for _, lines in pages for line in lines)
    raw_controls = sum(line.get("controls", 0) for _, lines in pages for line in lines)
    often, twice, fixed = _furniture(pages)
    ordered: list[dict] = []
    column_pages, dropped = [], 0
    for (height, lines), width in zip(pages, widths):
        body = []
        sizes_here = sorted(line["size"] for line in lines) or [0.0]
        usual = sizes_here[len(sizes_here) // 2]

        def furniture(other: dict, height: float = height) -> bool:
            return (
                other["rotated"]
                or (
                    _in_margin(other, height)
                    and (_key(other) in often or _key(other) in twice)
                )
                or _place(other) in fixed
            )

        for line in lines:
            key = _key(line)
            margin = _in_margin(line, height)
            numeral = (  # an entry number printed large, not a page number
                re.fullmatch(r"\d{1,3}\.?", line["text"])
                and line["size"] >= 1.8 * usual
            )
            if numeral and not line["rotated"]:
                body.append(line)
                continue
            if (
                line["rotated"]
                or (margin and (key in often or key in twice or key in {"", "0"}))
                or (
                    _place(line) in fixed
                    and not _word_in_text(line, lines, usual)
                    and not _entry_number(line, lines)
                )
            ) and not _title_start(line, lines, usual, furniture):
                dropped += 1
                continue
            body.append(line)
        if not body:
            continue
        page_lines = _merge_rows(
            _attach_numerals(
                _order(_join_words(_attach_glyphs(_drop_overprints(body))), width)
            )
        )
        if any(line["col"] for line in page_lines):
            column_pages.append(page_lines[0]["page"])
        rights: dict[int, list[float]] = defaultdict(list)
        for line in page_lines:
            rights[line["col"]].append(line["x1"])
        for line in page_lines:
            values = sorted(rights[line["col"]])
            line["right"] = values[int(len(values) * 0.9)]
            line["left"] = min(
                other["x0"] for other in page_lines if other["col"] == line["col"]
            )
        ordered.extend(page_lines)
    empty_pages = [
        number
        for number, (_, lines) in enumerate(pages, 1)
        if not any(not line["rotated"] for line in lines)
    ]

    sizes: Counter = Counter()
    gaps = []
    for prev, line in zip(ordered, ordered[1:]):
        sizes[line["size"]] += len(line["text"])
        if prev["page"] == line["page"] and prev["col"] == line["col"]:
            gap = line["y0"] - prev["y1"]
            if -2 < gap < line["size"] * 2:
                gaps.append(gap)
    body_size = sizes.most_common(1)[0][0] if sizes else 0.0
    usual_gap = statistics.median(gaps) if gaps else 0.0

    units: list[dict] = []
    current: dict | None = None
    prev: dict | None = None
    for line in ordered:
        start = current is None or prev is None
        if not start:
            ended = bool(END.search(prev["text"]))
            same_flow = prev["page"] == line["page"] and prev["col"] == line["col"]
            width = max(prev["right"] - prev["left"], 1.0)
            short = prev["x1"] < prev["right"] - 0.12 * width
            style = abs(line["size"] - prev["size"]) > 0.6 or (
                line["bold"] != prev["bold"] and (short or ended)
            )
            runs_on = (  # an unfinished sentence that continues in lower case
                not ended
                and line["text"][:1].islower()
                and abs(line["size"] - prev["size"]) <= 0.6
            )
            caps_lead = (  # a capitalised bold lead-in that runs over two lines
                prev["bold"]
                and line.get("lead_bold")
                and not ended
                and not short  # the lead reached the right edge and wrapped
                and abs(line["size"] - prev["size"]) <= 0.6
            )
            if MARKER.match(line["text"]) or line["bullet"]:
                start = True
            elif (
                caps_lead and same_flow and line["y0"] - prev["y1"] < 1.5 * line["size"]
            ):
                start = False
            elif runs_on and same_flow and line["y0"] - prev["y1"] < 1.5 * line["size"]:
                start = False  # e.g. a bold lead-in followed by its own sentence
            elif style:
                start = True
            elif not same_flow:
                start = ended or not line["text"][:1].islower() and short
            else:
                gap = line["y0"] - prev["y1"]
                start = (
                    gap > usual_gap + 0.45 * line["size"]
                    or gap < -line["size"]
                    or (ended and short)
                    or (ended and line["x0"] > prev["x0"] + 0.8 * line["size"])
                )
        if start:
            current = {
                "id": len(units) + 1,
                "page": line["page"],
                "page_end": line["page"],
                "col": line["col"],
                "size": line["size"],
                "bold": line["bold"],
                "bbox": [line["x0"], line["y0"], line["x1"], line["y1"]],
                "lines": 1,
                "bullet": line["bullet"],
                "text": line["text"],
            }
            units.append(current)
        else:
            text = current["text"]
            if re.search(r"\w[-‐­]$", text) and line["text"][:1].islower():
                current["text"] = text[:-1] + line["text"]
            else:
                current["text"] = text + " " + line["text"]
            current["lines"] += 1
            current["bold"] = current["bold"] and line["bold"]
            current["page_end"] = line["page"]
            if line["page"] == current["page"]:
                box = current["bbox"]
                current["bbox"] = [
                    min(box[0], line["x0"]),
                    min(box[1], line["y0"]),
                    max(box[2], line["x1"]),
                    max(box[3], line["y1"]),
                ]
        prev = line

    for unit in units:
        text = unit["text"]
        letters = [char for char in text if char.isalpha()]
        upper = len(letters) >= 4 and all(char.isupper() for char in letters)
        strong = (
            unit["size"] >= body_size + 1.0
            or (unit["bold"] and unit["size"] >= body_size - 0.2)
            or upper
        )
        if TOC.search(text):
            unit["kind"] = "toc"
            unit.pop("bullet")
        elif (
            strong
            and len(text) <= 200
            and unit["lines"] <= 4
            and not unit["bullet"]
            and not BULLET_START.match(text)
            and not (MARKER.match(text) and len(text) > 90)
        ):
            unit["kind"] = "heading"
            unit.pop("bullet")
        elif unit.pop("bullet") or MARKER.match(text):
            unit["kind"] = "item"
        else:
            unit["kind"] = "text"
        unit["bbox"] = [round(value, 1) for value in unit["bbox"]]

    chars = sum(len(unit["text"]) for unit in units)
    warnings = []
    if pages and chars / len(pages) < 250:
        warnings.append("low_text_needs_ocr")
    if empty_pages:
        warnings.append("pages_without_text")
    if raw_chars and raw_controls / raw_chars > 0.02:
        # a font without a usable character map: text comes out shifted
        # («/D\x03OLEHUDFLĂQ» for «La liberación»); it must be OCRed
        warnings.append("garbled_text_needs_ocr")
    if sum("�" in unit["text"] for unit in units) > 5:
        warnings.append("replacement_characters")
    return {
        "meta": {
            "version": VERSION,
            "pdf": path.name,
            "pdf_sha256": _sha256(path),
            "pages": len(pages),
            "units": len(units),
            "chars": chars,
            "body_size": body_size,
            "furniture_lines_dropped": dropped,
            "pages_without_text": empty_pages,
            "pages_with_columns": column_pages,
            "warnings": warnings,
        },
        "units": units,
    }


def save(result: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(
        json.dumps(result["meta"], ensure_ascii=False, indent=1), encoding="utf-8"
    )
    with (directory / "units.jsonl").open("w", encoding="utf-8") as stream:
        for unit in result["units"]:
            stream.write(json.dumps(unit, ensure_ascii=False) + "\n")


def load(directory: Path) -> dict:
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    with (directory / "units.jsonl").open(encoding="utf-8") as stream:
        units = [json.loads(line) for line in stream if line.strip()]
    return {"meta": meta, "units": units}


def _valid(directory: Path, version: str, sha: str | None = None) -> bool:
    meta_file = directory / "meta.json"
    if not (meta_file.exists() and (directory / "units.jsonl").exists()):
        return False
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    return meta.get("version") == version and (
        sha is None or meta.get("pdf_sha256") == sha
    )


def load_version(directory: Path, version: str) -> dict:
    """Units of one linearizer version. Each version is cached apart, so a new
    version never invalidates the answers of runs made with an older one."""
    if _valid(directory / "lin" / version, version):
        return load(directory / "lin" / version)
    if _valid(directory, version):  # layout used before versioned caches
        return load(directory)
    raise FileNotFoundError(
        f"No {version} units in {directory}; run `manifestos-batch prepare` with that version"
    )


def load_current(directory: Path) -> dict:
    """Units of the current version if prepared, else whatever the cache holds."""
    try:
        return load_version(directory, VERSION)
    except FileNotFoundError:
        return load(directory)


def cached(pdf: Path, directory: Path) -> dict:
    """Linearize once per PDF content and code version."""
    sha = _sha256(Path(pdf))
    target = directory / "lin" / VERSION
    if _valid(target, VERSION, sha):
        return load(target)
    if _valid(directory, VERSION, sha):
        return load(directory)
    result = linearize(pdf)
    save(result, target)
    if not (directory / "units.jsonl").exists():
        save(result, directory)
    return result
