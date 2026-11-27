"""`pledgetrace` command line: prepare, pair, extract, score, ceiling."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from . import backends, corpus, evaluate, infer, linearize

# Default layout of a project directory; every path can be set on the command line.
ORIGINALS = Path("data/pdfs")
PARSED = Path("data/annotations")
WORK = Path("runs")
CORPUS = Path("corpus.csv")


def _prepare_one(args: tuple[str, str]) -> tuple[str, dict | str]:
    pdf, work = Path(args[0]), Path(args[1])
    try:
        return pdf.stem, linearize.cached(pdf, work / pdf.stem)["meta"]
    except Exception as exc:  # one unreadable PDF must not stop the corpus
        return pdf.stem, f"ERROR {type(exc).__name__}: {exc}"


def cmd_prepare(args: argparse.Namespace) -> int:
    if not args.originals.is_dir():
        print(f"PDF directory not found: {args.originals}", file=sys.stderr)
        return 2
    pdfs = sorted(args.originals.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files in {args.originals}", file=sys.stderr)
        return 2
    if args.only:
        pdfs = [pdf for pdf in pdfs if pdf.stem in args.only]
    jobs = [(str(pdf), str(args.work)) for pdf in pdfs]
    failed = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for stem, meta in pool.map(_prepare_one, jobs):
            if isinstance(meta, str):
                failed += 1
                print(f"{stem}\t{meta}")
            else:
                print(
                    f"{stem}\t{meta['pages']}p\t{meta['units']}u\t"
                    f"{','.join(meta['warnings']) or 'ok'}"
                )
    return 1 if failed else 0


def _linearized(work: Path) -> dict[str, dict]:
    return {
        directory.name: linearize.load_current(directory)
        for directory in sorted(work.iterdir())
        if (directory / "units.jsonl").exists()
    }


def cmd_pair(args: argparse.Namespace) -> int:
    previous = (
        {row["doc_id"]: row["split"] for row in corpus.read(args.corpus)}
        if args.corpus.exists()
        else {}
    )
    rows = corpus.pair(_linearized(args.work), args.parsed)
    corpus.assign_split(rows, previous)
    corpus.write(args.corpus, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    print(json.dumps(counts, ensure_ascii=False))
    return 0


def _documents(args: argparse.Namespace) -> list[dict]:
    rows = [row for row in corpus.read(args.corpus) if row["pdf"]]
    gone = [row["doc_id"] for row in rows if not (args.originals / row["pdf"]).exists()]
    if gone:
        print(
            f"WARNING: {len(gone)} documents in {args.corpus} point to a PDF that no longer "
            f"exists ({', '.join(gone[:6])}{'…' if len(gone) > 6 else ''}). They are skipped. "
            "Run `pledgetrace prepare` and `pledgetrace pair` to "
            "actualizar el emparejamiento."
        )
        rows = [row for row in rows if row["doc_id"] not in gone]
    if getattr(args, "doc", None):
        rows = [row for row in rows if row["doc_id"] in args.doc]
    elif getattr(args, "split", None):
        rows = [row for row in rows if row["split"] == args.split]
    return rows


def cmd_ceiling(args: argparse.Namespace) -> int:
    out = []
    for row in _documents(args):
        if not row["reference"]:
            continue
        units = linearize.load_current(args.work / Path(row["pdf"]).stem)["units"]
        result = evaluate.ceiling(
            units, evaluate.read_rows(args.parsed / row["reference"])
        )
        out.append({"doc_id": row["doc_id"], "split": row["split"], **result})
        print(
            f"{row['doc_id']:<22}{row['split']:<6}rows={result['reference_rows']:<5}"
            f"located={result['located_share']}  by_unit={result['unit_share']}  "
            f"by_sentence={result['sentence_share']}"
        )
    if args.output:
        args.output.write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    return 0


def _run_dir(args: argparse.Namespace, row: dict) -> Path:
    return args.work / Path(row["pdf"]).stem / "runs" / args.run


def _answer(backend, units: list[dict], chunk: dict, folder: Path, retries: int) -> str:
    """Ask one chunk; keep every raw answer. Returns '' or an error message."""
    target = folder / f"{chunk['id']}.response.json"
    prompt = (folder / f"{chunk['id']}.prompt.md").read_text(encoding="utf-8")
    error = ""
    for attempt in range(retries + 1):
        user = (
            prompt
            if not error
            else (
                f"{prompt}\n\nTu respuesta anterior no era válida: {error}\nCorrígela."
            )
        )
        try:
            raw = backend.complete(infer.system_prompt(), user, attempt)
        except Exception as exc:  # network/model failure: leave chunk pending
            return f"{type(exc).__name__}: {exc}"
        (folder / f"{chunk['id']}.raw{attempt}.txt").write_text(raw, encoding="utf-8")
        try:
            infer.parse(raw, units, chunk)
        except ValueError as exc:
            error = str(exc)
            continue
        target.write_text(raw, encoding="utf-8")
        return ""
    for attempt in range(
        retries + 1
    ):  # last resort: the first answer Python can repair
        raw = (folder / f"{chunk['id']}.raw{attempt}.txt").read_text(encoding="utf-8")
        try:
            infer.parse(raw, units, chunk, repair=True)
        except ValueError:
            continue
        break
    else:
        return f"invalid after {retries + 1} attempts: {error}"
    target.write_text(raw, encoding="utf-8")
    (folder / f"{chunk['id']}.repaired").write_text(error, encoding="utf-8")
    return ""


def cmd_extract(args: argparse.Namespace) -> int:
    infer.PARTS = args.parts
    backend = backends.make(args.backend, args.model, args.host, args.num_ctx)
    status = 0
    for row in _documents(args):
        stem = Path(row["pdf"]).stem
        planned = _run_dir(args, row) / "chunks" / "plan.json"
        version = (
            json.loads(planned.read_text(encoding="utf-8")).get("linearize")
            if planned.exists()
            else None
        )
        if version and version != linearize.VERSION:
            data = linearize.load_version(args.work / stem, version)  # keep its units
        else:
            data = linearize.cached(args.originals / row["pdf"], args.work / stem)
        units, meta = data["units"], data["meta"]
        if {"low_text_needs_ocr", "garbled_text_needs_ocr"} & set(meta["warnings"]):
            print(
                f"{row['doc_id']}: text layer not usable without OCR ({', '.join(meta['warnings'])}); skipped"
            )
            status = 2
            continue
        directory = _run_dir(args, row)
        try:
            chunks = infer.write_prompts(directory, units, meta, args.chunk_chars)
        except ValueError as exc:
            print(f"{row['doc_id']}: {exc}")
            status = 2
            continue
        folder = directory / "chunks"
        pending = [
            c for c in chunks if not (folder / f"{c['id']}.response.json").exists()
        ]
        for chunk in pending:  # an earlier raw answer may be valid under current rules
            for raw in sorted(folder.glob(f"{chunk['id']}.raw*.txt")):
                try:
                    infer.parse(raw.read_text(encoding="utf-8"), units, chunk)
                except ValueError:
                    continue
                (folder / f"{chunk['id']}.response.json").write_bytes(raw.read_bytes())
                break
            else:  # every attempt was spent: the first answer Python can repair
                raws = sorted(folder.glob(f"{chunk['id']}.raw*.txt"))
                if len(raws) > args.retries:
                    for raw in raws:
                        try:
                            infer.parse(
                                raw.read_text(encoding="utf-8"), units, chunk, True
                            )
                        except ValueError:
                            continue
                        (folder / f"{chunk['id']}.response.json").write_bytes(
                            raw.read_bytes()
                        )
                        (folder / f"{chunk['id']}.repaired").write_text(
                            "recovered", encoding="utf-8"
                        )
                        break
        pending = [
            c for c in chunks if not (folder / f"{c['id']}.response.json").exists()
        ]
        if backend and pending:
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                errors = list(
                    pool.map(
                        lambda c: _answer(backend, units, c, folder, args.retries),
                        pending,
                    )
                )
            for chunk, error in zip(pending, errors):
                if error:
                    print(f"{row['doc_id']} {chunk['id']}: {error}")
            pending = [
                c for c in chunks if not (folder / f"{c['id']}.response.json").exists()
            ]
        labels, invalid = [], []
        for chunk in chunks:
            target = folder / f"{chunk['id']}.response.json"
            if not target.exists():
                continue
            try:
                labels.append(
                    infer.parse(
                        target.read_text(encoding="utf-8"),
                        units,
                        chunk,
                        repair=(folder / f"{chunk['id']}.repaired").exists(),
                    )
                )
            except ValueError as exc:
                invalid.append(f"{chunk['id']}: {exc}")
        if pending or invalid:
            status = 2
            print(
                f"{row['doc_id']}: {len(pending)} of {len(chunks)} chunks pending in {folder}"
                + "".join(f"\n  INVALID {item}" for item in invalid)
            )
            continue
        numbered = infer.apply_enumeration(units, labels)
        ranges_split = infer.split_merged_ranges(units, labels)
        rows = infer.build_rows(units, labels, meta)
        summary = (
            [False] * len(rows) if args.keep_summaries else infer.summary_blocks(rows)
        )
        rows = [row for row, drop in zip(rows, summary) if not drop]
        preamble = infer.preamble_units(units)
        kept = [row for row in rows if int(row["unit_from"]) not in preamble]
        # A large drop means the numbering is not one running count of proposals
        # (e.g. it restarts in every chapter): then the rule does not apply.
        preamble_dropped = len(rows) - len(kept)
        if preamble_dropped <= 0.15 * len(rows):
            rows = kept
        else:
            preamble_dropped = 0
        with (directory / "proposals.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=infer.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        # A re-export (backend "files") keeps the model that answered the chunks.
        previous = {}
        if (directory / "run.json").exists():
            previous = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        answered_by = {"backend": args.backend, "model": args.model}
        if args.backend == "files" and previous.get("backend") not in (None, "files"):
            answered_by = {
                "backend": previous["backend"],
                "model": previous.get("model"),
                "reexported": True,
            }
        (directory / "run.json").write_text(
            json.dumps(
                {
                    "doc_id": row["doc_id"],
                    "pdf": row["pdf"],
                    "pdf_sha256": meta["pdf_sha256"],
                    "linearize_version": meta["version"],
                    "prompt_version": infer.version(),
                    **answered_by,
                    "num_ctx": getattr(args, "num_ctx", None),
                    "chunk_chars": args.chunk_chars,
                    "chunks": len(chunks),
                    "proposals": len(rows),
                    "summary_rows_dropped": sum(summary),
                    "preamble_rows_dropped": preamble_dropped,
                    "ranges_split_as_separate_measures": ranges_split,
                    "proposals_set_by_document_numbering": numbered,
                    "long_headings_kept_as_proposals": sum(
                        label["promoted"] for label in labels
                    ),
                    "ranges_split_by_python": sum(
                        label["repaired"] for label in labels
                    ),
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        if args.publish:
            args.publish.mkdir(parents=True, exist_ok=True)
            (args.publish / f"{row['doc_id']}.csv").write_bytes(
                (directory / "proposals.csv").read_bytes()
            )
        print(
            f"{row['doc_id']}: {len(rows)} proposals -> {directory / 'proposals.csv'}"
        )
    return status


def cmd_score(args: argparse.Namespace) -> int:
    totals = {"strict": 0, "pred": 0, "ref": 0, "recalled": 0.0, "supported": 0.0}
    for row in _documents(args):
        directory = _run_dir(args, row)
        if not row["reference"] or not (directory / "proposals.csv").exists():
            continue
        plan = json.loads(
            (directory / "chunks" / "plan.json").read_text(encoding="utf-8")
        )
        units = linearize.load_version(
            args.work / Path(row["pdf"]).stem, plan.get("linearize", "linearize-7")
        )["units"]
        with (directory / "proposals.csv").open(newline="", encoding="utf-8") as stream:
            predicted = list(csv.DictReader(stream))
        try:
            result = evaluate.score(
                units, evaluate.read_rows(args.parsed / row["reference"]), predicted
            )
        except KeyError as exc:  # the CSV does not match these units
            print(
                f"{row['doc_id']}: skipped, the CSV does not match its units ({exc})"
            )
            continue
        (directory / "eval.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        totals["strict"] += result["strict_matches"]
        totals["pred"] += result["predicted"]
        totals["ref"] += result["reference_located"]
        totals["recalled"] += (result["recall_text"] or 0) * result["reference_located"]
        totals["supported"] += (result["precision_text"] or 0) * result["predicted"]
        print(
            f"{row['doc_id']:<20}{row['split']:<5} ref={result['reference_located']:<5}"
            f"pred={result['predicted']:<5} P_text={result['precision_text']} "
            f"R_text={result['recall_text']} | exact P={result['strict_precision']} "
            f"R={result['strict_recall']} F1={result['strict_f1']} "
            f"| sec1={result['seccion1_agreement']}"
        )
    if totals["pred"] and totals["ref"]:
        print(
            f"TOTAL pred={totals['pred']} ref={totals['ref']} "
            f"P_text={totals['supported'] / totals['pred']:.3f} "
            f"R_text={totals['recalled'] / totals['ref']:.3f} "
            f"exact P={totals['strict'] / totals['pred']:.3f} R={totals['strict'] / totals['ref']:.3f}"
        )
    return 0


def build() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="pledgetrace", description=__doc__)
    root.add_argument(
        "--pdfs",
        "--originals",
        dest="originals",
        metavar="DIR",
        type=Path,
        default=ORIGINALS,
        help="PDF directory (default: %(default)s)",
    )
    root.add_argument(
        "--annotations",
        "--parsed",
        dest="parsed",
        metavar="DIR",
        type=Path,
        default=PARSED,
        help="Directory of manual annotations, optional (default: %(default)s)",
    )
    root.add_argument("--work", type=Path, default=WORK)
    root.add_argument("--corpus", type=Path, default=CORPUS)
    sub = root.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="PDF -> numbered units (cached, no model)")
    prepare.add_argument("--only", nargs="*", help="PDF stems")
    prepare.add_argument("--jobs", type=int, default=4)
    prepare.set_defaults(run_command=cmd_prepare)

    pair = sub.add_parser("pair", help="Pair PDFs with references; fix dev/test split")
    pair.set_defaults(run_command=cmd_pair)

    ceiling = sub.add_parser("ceiling", help="Unit/reference agreement, no model")
    ceiling.add_argument("--doc", nargs="+", action="extend", help="Document ids; repeat or list several")
    ceiling.add_argument("--split", choices=["dev", "test"])
    ceiling.add_argument("--output", type=Path)
    ceiling.set_defaults(run_command=cmd_ceiling)

    extract = sub.add_parser("extract", help="Label units with a model and export CSV")
    extract.add_argument("--doc", nargs="+", action="extend", help="Document ids; repeat or list several")
    extract.add_argument("--split", choices=["dev", "test", "unlabelled"])
    extract.add_argument("--run", required=True, help="Run name, e.g. the model used")
    extract.add_argument(
        "--backend", choices=["files", "ollama", "openai", "anthropic"], default="files"
    )
    extract.add_argument("--model")
    extract.add_argument("--host")
    extract.add_argument(
        "--keep-summaries",
        action="store_true",
        help="Do not drop proposals of an opening summary that the body repeats",
    )
    extract.add_argument(
        "--parts",
        choices=["none", "prose"],
        default="none",
        help="none: units are cited whole (default); prose: paragraphs may be cited by sentence",
    )
    extract.add_argument(
        "--num-ctx", type=int, default=16384, help="Ollama context window in tokens"
    )
    extract.add_argument("--chunk-chars", type=int, default=14000)
    extract.add_argument("--jobs", type=int, default=1)
    extract.add_argument("--retries", type=int, default=2)
    extract.add_argument("--publish", type=Path, help="Also copy finished CSVs here")
    extract.set_defaults(run_command=cmd_extract)

    score = sub.add_parser("score", help="Compare a run with the manual reference")
    score.add_argument("--doc", nargs="+", action="extend", help="Document ids; repeat or list several")
    score.add_argument("--split", choices=["dev", "test"])
    score.add_argument("--run", required=True)
    score.set_defaults(run_command=cmd_score)
    return root


def main(argv: list[str] | None = None) -> int:
    args = build().parse_args(argv)
    return args.run_command(args)


if __name__ == "__main__":
    sys.exit(main())
