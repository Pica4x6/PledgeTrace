# Using PledgeTrace

## Project layout

PledgeTrace works inside a project directory. By default it expects:

```
data/pdfs/          manifestos in PDF
data/annotations/   manual annotations, one CSV per manifesto (optional)
corpus.csv          document registry, written by `pair`
runs/               reconstructed documents and extraction runs
```

Every path can be changed with the global options, which go before the command:

```sh
pledgetrace --pdfs my/pdfs --annotations my/csv --work my/runs --corpus my/corpus.csv prepare
```

## Commands

### `prepare`: reconstruct the documents

```sh
pledgetrace prepare [--only STEM ...] [--jobs 4]
```

Converts each PDF into numbered text units and stores them in
`runs/<stem>/lin/<linearizer-version>/` (`units.jsonl` and `meta.json`). No model is used.
Results are cached by PDF content and linearizer version, so running it again is cheap
and runs made with an older version keep their units. Documents whose text layer is
missing or garbled are flagged in `meta.json` and skipped by `extract`.

### `pair`: register the corpus

```sh
pledgetrace pair
```

Writes `corpus.csv`, with one row per document. Each annotation file is matched to the
PDF whose text contains it, among the PDFs whose name ends with the same election
suffix (for example `IU_2011.csv` and `33220_2011.pdf`). Annotated documents are
assigned to a development or test split; PDFs without annotation are `unlabelled`.
Splits already present in `corpus.csv` are kept, so the partition is fixed the first
time and stays stable when documents are added. You can edit the `split` column by hand.

### `extract`: select and export proposals

```sh
pledgetrace extract --run NAME --backend ollama --model MODEL [options]
```

| Option | Meaning |
|---|---|
| `--run NAME` | Name of the run; use one name per model and configuration |
| `--backend` | `ollama`, `openai`, `anthropic` or `files` |
| `--model`, `--host` | Model name and server address |
| `--doc ID ...` / `--split dev` | Restrict to some documents or a split |
| `--num-ctx 16384` | Ollama context window |
| `--chunk-chars 14000` | Maximum size of each chunk sent to the model |
| `--retries 2` | New attempts after an invalid answer |
| `--publish DIR` | Also copy each finished CSV to `DIR/<doc_id>.csv` |

Each document is divided into chunks of consecutive units. For each chunk, the prompt,
every raw answer and the accepted answer are stored in
`runs/<stem>/runs/<run>/chunks/`. Invalid answers (unknown identifiers, overlapping
ranges, malformed JSON) are retried; the first attempt uses temperature 0 and retries
sample with a fixed seed. A chunk that stays invalid is left pending and reported; it
is never treated as a chunk without proposals. Running the same command again only
asks for pending chunks.

Backends:

- `ollama`: local models served by Ollama (`--host http://localhost:11434`).
- `openai`: any OpenAI-compatible `/chat/completions` endpoint; key in `OPENAI_API_KEY`.
- `anthropic`: Anthropic API; key in `ANTHROPIC_API_KEY`.
- `files`: no model call. Prompts are written to disk and answers are read from
  `chunks/<id>.response.json`, so they can be produced by any other means. It is also
  how a run is re-exported after a change in the post-processing rules.

The result of each document is `runs/<stem>/runs/<run>/proposals.csv` plus
`run.json` with the configuration (model, backend, linearizer and prompt versions,
context, chunk size).

### `score`: compare with manual annotations

```sh
pledgetrace score --run NAME [--split test]
```

Locates each annotated proposal in the reconstructed text and compares it with the
extracted proposals, without calling the model. It reports:

- **interval precision, recall and F1**: an extracted proposal matches an annotated one
  when their character intervals overlap with intersection over union of at least 0.8
  (one-to-one);
- **text precision and recall**: share of extracted proposals mostly covered by
  annotations, and of annotations mostly covered by extracted proposals.

Annotations that cannot be located in the text are counted separately and left out of
recall. Results per document are written to `eval.json` in the run directory.

### `ceiling`: check the units against the annotations

```sh
pledgetrace ceiling
```

Reports, per document, how many annotated proposals can be located in the text and how
many start and end at unit or sentence boundaries. It separates reconstruction problems
from selection problems and needs no model.

## Annotation format

One CSV per manifesto, UTF-8, with at least the column `propuesta_texto` (one proposal
per row, copied from the document). Optional columns: `seccion1`, `seccion2`,
`seccion3` and `propuesta_id`. Rows with empty text are ignored.
