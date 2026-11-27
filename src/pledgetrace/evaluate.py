"""Evaluation against a manual reference, in source-text coordinates.

Reference rows and predicted proposals are both mapped to character spans of the
normalised document text, so scores do not depend on fuzzy string matching
between two long lists. Reference rows that cannot be found in the PDF text are
reported and left out of the scores (they may be annotation edits, or text the
PDF layer does not contain).
"""

from __future__ import annotations

import bisect
import csv
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from .segment import sentences
from .textnorm import norm

Span = tuple[int, int]


def read_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig", errors="replace") as stream:
        return [
            row
            for row in csv.DictReader(stream)
            if (row.get("propuesta_texto") or "").strip()
        ]


def source_index(units: list[dict]) -> tuple[str, list[Span]]:
    parts, spans, position = [], [], 0
    for unit in units:
        text = norm(unit["text"])
        spans.append((position, position + len(text)))
        parts.append(text)
        position += len(text)
    return "".join(parts), spans


def _fuzzy(document: str, text: str, seed: int = 24) -> Span | None:
    if len(text) < seed * 2:
        return None
    votes: Counter = Counter()
    total = 0
    for offset in range(0, len(text) - seed + 1, seed):
        total += 1
        piece = text[offset : offset + seed]
        position = document.find(piece)
        hits = 0
        while position != -1 and hits < 8:
            votes[(position - offset) // 60] += 1
            hits += 1
            position = document.find(piece, position + 1)
    if not votes:
        return None
    bucket, count = votes.most_common(1)[0]
    count += (
        votes.get(bucket + 1, 0)
        if votes.get(bucket + 1, 0)
        else votes.get(bucket - 1, 0)
    )
    if count < max(2, 0.4 * total):
        return None
    start = max(0, bucket * 60)
    return start, min(len(document), start + len(text))


def locate(document: str, texts: list[str]) -> list[Span | None]:
    """Span of each reference text in the document, preferring reading order."""
    spans: list[Span | None] = []
    cursor = 0
    for raw in texts:
        text = norm(raw)
        span = None
        if len(text) >= 12:
            position = document.find(text, max(0, cursor - 200))
            if position == -1:
                position = document.find(text)
            if position != -1:
                span = (position, position + len(text))
            else:
                span = _fuzzy(document, text)
        if span:
            cursor = span[1]
        spans.append(span)
    return spans


def _overlap(a: Span, b: Span) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _iou(a: Span, b: Span) -> float:
    inter = _overlap(a, b)
    return inter / ((a[1] - a[0]) + (b[1] - b[0]) - inter) if inter else 0.0


def _covered(span: Span, others: list[Span]) -> float:
    """Share of span covered by the (sorted, possibly overlapping) other spans."""
    total, edge = 0, span[0]
    for other in others:
        if other[1] <= edge:
            continue
        if other[0] >= span[1]:
            break
        start = max(other[0], edge)
        end = min(other[1], span[1])
        if end > start:
            total += end - start
            edge = end
    return total / max(1, span[1] - span[0])


def _boundaries(
    units: list[dict], unit_spans: list[Span]
) -> tuple[list[int], list[int]]:
    """Normalised offsets where a unit or one of its sentence parts starts."""
    coarse, fine = [], []
    for unit, (start, _) in zip(units, unit_spans):
        coarse.append(start)
        for part_start, _ in sentences(unit["text"]):
            fine.append(start + len(norm(unit["text"][:part_start])))
    return coarse, fine


def _near(value: int, sorted_values: list[int], tolerance: int) -> bool:
    index = bisect.bisect_left(sorted_values, value - tolerance)
    return index < len(sorted_values) and sorted_values[index] <= value + tolerance


def ceiling(units: list[dict], reference: list[dict]) -> dict:
    """Share of reference rows whose limits fall on unit (or sentence) limits.

    This is the best exact-boundary recall any labeller of unit IDs can reach.
    """
    document, unit_spans = source_index(units)
    located = locate(document, [row["propuesta_texto"] for row in reference])
    coarse, fine = _boundaries(units, unit_spans)
    coarse_ends = sorted(coarse[1:] + [len(document)])
    fine_ends = sorted(fine[1:] + [len(document)])
    found = by_unit = by_sentence = 0
    for span in located:
        if span is None:
            continue
        found += 1
        tolerance = max(8, int(0.04 * (span[1] - span[0])))
        if _near(span[0], coarse, tolerance) and _near(span[1], coarse_ends, tolerance):
            by_unit += 1
        if _near(span[0], fine, tolerance) and _near(span[1], fine_ends, tolerance):
            by_sentence += 1
    rows = len(reference)
    return {
        "reference_rows": rows,
        "located": found,
        "located_share": round(found / rows, 4) if rows else None,
        "unit_share": round(by_unit / found, 4) if found else None,
        "sentence_share": round(by_sentence / found, 4) if found else None,
    }


def _same_section(reference: dict, predicted: dict) -> bool:
    """Reference seccion1 found among the predicted section titles (or vice versa)."""
    wanted = norm(reference.get("seccion1", ""))
    if not wanted:
        return False
    for key in ("seccion1", "seccion2", "seccion3"):
        got = norm(predicted.get(key, ""))
        if not got:
            continue
        if (
            wanted in got
            or got in wanted
            or SequenceMatcher(None, wanted, got).ratio() >= 0.8
        ):
            return True
    return False


def score(units: list[dict], reference: list[dict], predicted: list[dict]) -> dict:
    """Precision/recall of predicted proposals (rows with unit_from/unit_to)."""
    document, unit_spans = source_index(units)
    by_id = {unit["id"]: span for unit, span in zip(units, unit_spans)}
    ref_spans = locate(document, [row["propuesta_texto"] for row in reference])
    refs = [(span, row) for span, row in zip(ref_spans, reference) if span]
    text_of = {unit["id"]: unit["text"] for unit in units}
    preds = []
    for row in predicted:
        first, last = int(row["unit_from"]), int(row["unit_to"])
        start = by_id[first][0] + len(norm(text_of[first][: int(row["char_from"])]))
        end = by_id[last][0] + len(norm(text_of[last][: int(row["char_to"])]))
        preds.append(((start, end), row))
    ref_sorted = sorted(span for span, _ in refs)
    pred_sorted = sorted(span for span, _ in preds)
    recalled = [(_covered(span, pred_sorted) >= 0.8) for span, _ in refs]
    supported = [(_covered(span, ref_sorted) >= 0.5) for span, _ in preds]
    strict = 0
    section_ok = section_total = 0
    missed, spurious = [], []
    used = set()
    for index, (span, row) in enumerate(refs):
        best, best_iou = None, 0.0
        for other_index, (other, _) in enumerate(preds):
            if other[0] >= span[1]:
                continue
            if other[1] <= span[0] or other_index in used:
                continue
            value = _iou(span, other)
            if value > best_iou:
                best, best_iou = other_index, value
        if best is not None and best_iou >= 0.8:
            used.add(best)
            strict += 1
            section_total += 1
            if _same_section(row, preds[best][1]):
                section_ok += 1
        if not recalled[index]:
            missed.append(row["propuesta_texto"][:160])
    for flag, (_, row) in zip(supported, preds):
        if not flag:
            spurious.append(f"[{row['unit_from']}] {row['propuesta_texto'][:150]}")

    def ratio(a: int, b: int) -> float | None:
        return round(a / b, 4) if b else None

    precision, recall = ratio(strict, len(preds)), ratio(strict, len(refs))
    return {
        "reference_rows": len(reference),
        "reference_located": len(refs),
        "predicted": len(preds),
        "recall_text": ratio(sum(recalled), len(refs)),
        "precision_text": ratio(sum(supported), len(preds)),
        "strict_matches": strict,
        "strict_precision": precision,
        "strict_recall": recall,
        "strict_f1": round(2 * precision * recall / (precision + recall), 4)
        if precision and recall
        else 0.0,
        "seccion1_agreement": ratio(section_ok, section_total),
        "missed_examples": missed[:25],
        "unsupported_examples": spurious[:25],
        "missed": len(missed),
        "unsupported": len(spurious),
    }
