# PledgeTrace

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24590059.svg)](https://doi.org/10.5281/zenodo.24590059)

**Traceable extraction of policy proposals from electoral manifestos.**

PledgeTrace turns a manifesto in PDF into a table of policy proposals. Each proposal is
assembled from the extracted document text and retains its source location: page, text
units, character offsets and bounding box.

It splits the work in three:

1. **Python reconstructs the document.** The PDF is converted into an ordered sequence
   of numbered text units: reading order across columns, running headers and page
   numbers removed, list markers and printed proposal numbers attached to their text.
2. **A language model selects.** It sees the numbered units and answers only with the
   identifiers of headings and proposals, for example `{"h": [14], "p": ["13", "16-17"]}`.
   It never writes the proposal text.
3. **Python rebuilds the proposals.** The selected units are turned into proposal
   records, with deterministic rules for numbered lists, paragraph boundaries and
   introductory text.

Every intermediate artifact (units, prompts, raw responses, run metadata) is stored, so
results can be inspected, compared across models and re-exported without calling the
model again.

## Installation

PledgeTrace needs Python 3.11 or later. With [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/Pica4x6/PledgeTrace.git
cd PledgeTrace
uv sync
uv run pledgetrace --help
```

or with pip: `pip install .`

To use a local model, install [Ollama](https://ollama.com) and pull a model, for example
`ollama pull qwen3:14b`. Remote OpenAI-compatible and Anthropic endpoints are also
supported (see [docs/usage.md](docs/usage.md)).

## Quick start

Put your PDFs in `data/pdfs/` and, if you have them, manual annotations in
`data/annotations/` (optional; see [docs/usage.md](docs/usage.md) for the format). Then:

```sh
uv run pledgetrace prepare                 # PDFs -> numbered text units (no model)
uv run pledgetrace pair                    # register documents in corpus.csv
uv run pledgetrace extract --run qwen14b \
    --backend ollama --model qwen3:14b     # select proposals and export CSVs
uv run pledgetrace score --run qwen14b     # compare with annotations (optional)
```

The proposals of each document are written to
`runs/<pdf-name>/runs/<run>/proposals.csv`.

The manifestos used as examples in the PledgeTrace paper, and a script to process them,
are in [examples/paper](examples/paper). The PDFs themselves are not
included; [examples/paper](examples/paper/README.md#getting-the-pdfs) explains how to
obtain them from the [Manifesto Project](https://manifesto-project.wzb.eu).

## Output

Each row of `proposals.csv` is one proposal:

| Column | Content |
|---|---|
| `seccion1`–`seccion3` | Section headings above the proposal |
| `propuesta_id` | Number printed in the document, when there is one |
| `propuesta_texto` | Proposal text, assembled from the reconstructed units |
| `pagina`, `pagina_fin` | First and last page |
| `unit_from`, `unit_to` | First and last text unit |
| `char_from`, `char_to` | Character offsets inside those units |
| `bbox` | Bounding box of the first unit on its page |
| `pdf` | Source file |

## Requirements and limitations

- PDFs need a usable text layer. Scanned documents must go through OCR first; documents
  without usable text are reported and skipped.
- The prompt is written in Spanish and the tool has been evaluated on Spanish manifestos
  in Spanish, Catalan, Galician and Basque. Other languages and countries should be
  evaluated before use.
- Reconstruction is heuristic: unusual layouts, tables or typography can still produce
  errors. Results are meant to be reviewed, and every row can be checked against its page.

## Versions and reproducibility

PledgeTrace follows [semantic versioning](https://semver.org). Each release is tagged
(`v1.0.0`, …) and listed in [CHANGELOG.md](CHANGELOG.md). Reconstructed documents are
cached per linearizer version (`linearize-40` in 1.0.0), and every run records the
version it used, so results remain reproducible after the code evolves. To reproduce a
published analysis, check out the release it cites:

```sh
git checkout v1.0.0
uv sync
```

## Citation

If you use PledgeTrace, please cite the release you used. Version 1.0.0 is archived on
Zenodo: [doi:10.5281/zenodo.24590059](https://doi.org/10.5281/zenodo.24590059).

## License

Copyright © 2026 Pablo Picazo-Sanchez and Lara Ortiz-Martin.

PledgeTrace is free software under the GNU Affero General Public License v3 or later
(see [LICENSE](LICENSE)), the same family of license as its PDF library
[PyMuPDF](https://pymupdf.readthedocs.io). The license does not cover the manifestos,
manual annotations or model weights you use with it. The design builds on ideas from
[ParlaIbero](https://github.com/rodrodr/parlaibero-tools).
