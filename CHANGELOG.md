# Changelog

All notable changes to PledgeTrace are listed here. Versions follow
[semantic versioning](https://semver.org): a new major version may change outputs or
the command line; minor versions add features; patch versions fix bugs.

When a change alters the reconstructed text, the linearizer version is increased. Runs
record the linearizer version they used, and older reconstructions stay cached.

## [1.0.0] - 2026-11-27

First public release, described in the SoftwareX paper.

- Command line `pledgetrace` with `prepare`, `pair`, `extract`, `score` and `ceiling`.
- Document reconstruction `linearize-40`: reading order across columns, removal of
  running headers and page numbers, hidden text under opaque boxes, list markers and
  printed proposal numbers, ligatures, and detection of PDFs that need OCR.
- Selection by identifiers with Ollama, OpenAI-compatible, Anthropic and file-based
  backends; validation and retries (greedy first attempt, seeded sampling on retries).
- Deterministic post-processing: document numbering, paragraph-range splitting,
  preamble, table-of-contents and repeated-summary filters.
- Evaluation by character intervals (IoU ≥ 0.8) and text coverage.
