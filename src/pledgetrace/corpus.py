"""Corpus table: which reference CSV belongs to which PDF, and the dev/test split.

PDFs are named by Manifesto Project party code and references by party name, so
the pairing is found from content: a reference belongs to the same-election PDF
whose text contains most of its rows. The split is fixed the first time the table
is written and never reassigned afterwards.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from .evaluate import read_rows, source_index
from .textnorm import norm

FIELDS = [
    "doc_id",
    "pdf",
    "reference",
    "party",
    "election",
    "split",
    "pair_score",
    "runner_up",
    "reference_rows",
    "pages",
    "units",
    "warnings",
]
# Documents that must always be in the development set (e.g. those inspected
# while designing prompts or rules). Empty by default; the split of documents
# already in corpus.csv is kept, so a project fixes its split once.
SEEN: frozenset[str] = frozenset()


def _election(stem: str) -> str:
    tail = stem.rsplit("_", 1)[-1]
    return tail if tail.isdigit() else ""


def pair(linearized: dict[str, dict], parsed: Path, sample: int = 80) -> list[dict]:
    """linearized: pdf stem -> {meta, units}."""
    documents = {
        stem: source_index(item["units"])[0] for stem, item in linearized.items()
    }
    rows_out, used = [], set()
    for reference in sorted(Path(parsed).glob("*.csv")):
        rows = read_rows(reference)
        texts = [norm(row["propuesta_texto"]) for row in rows]
        texts = [text for text in texts if len(text) >= 40]
        step = max(1, len(texts) // sample)
        probe = texts[::step][:sample]
        election = _election(reference.stem)
        scores = []
        for stem, document in documents.items():
            if _election(stem) != election:
                continue
            hits = sum(1 for text in probe if text in document)
            scores.append((hits / len(probe) if probe else 0.0, stem))
        scores.sort(reverse=True)
        best = scores[0] if scores else (0.0, "")
        second = scores[1] if len(scores) > 1 else (0.0, "")
        matched = best[1] if best[0] >= 0.3 else ""
        used.add(matched)
        meta = linearized[matched]["meta"] if matched else {}
        rows_out.append(
            {
                "doc_id": reference.stem,
                "pdf": f"{matched}.pdf" if matched else "",
                "reference": reference.name,
                "party": reference.stem.rsplit("_", 1)[0],
                "election": election,
                "split": "",
                "pair_score": round(best[0], 3),
                "runner_up": f"{second[1]}:{second[0]:.2f}" if second[1] else "",
                "reference_rows": len(rows),
                "pages": meta.get("pages", ""),
                "units": meta.get("units", ""),
                "warnings": ";".join(meta.get("warnings", [])),
            }
        )
    for stem, item in sorted(linearized.items()):
        if stem in used:
            continue
        meta = item["meta"]
        rows_out.append(
            {
                "doc_id": stem,
                "pdf": f"{stem}.pdf",
                "reference": "",
                "party": "",
                "election": _election(stem),
                "split": "unlabelled",
                "pair_score": "",
                "runner_up": "",
                "reference_rows": "",
                "pages": meta["pages"],
                "units": meta["units"],
                "warnings": ";".join(meta["warnings"]),
            }
        )
    return rows_out


def assign_split(rows: list[dict], previous: dict[str, str]) -> None:
    """dev/test is kept for ever once given; the other labels follow the files."""
    for row in rows:
        before = previous.get(row["doc_id"], "")
        if not row["reference"]:
            row["split"] = "unlabelled"
        elif not row["pdf"]:
            row["split"] = "no_pdf"
        elif before in ("dev", "test"):
            row["split"] = before
        elif row["doc_id"] in SEEN:
            row["split"] = "dev"
        else:
            digest = hashlib.sha1(row["doc_id"].encode()).digest()
            row["split"] = "dev" if digest[0] % 5 == 0 else "test"
    # Two references on one PDF (a coalition programme): if one was used for
    # development the PDF has been seen, so the other cannot be a blind test.
    seen = {row["pdf"] for row in rows if row["split"] == "dev"}
    for row in rows:
        if row["split"] == "test" and row["pdf"] in seen:
            row["split"] = "dev"


def read(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
