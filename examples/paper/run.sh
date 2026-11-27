#!/bin/sh
# Examples of the PledgeTrace paper: Izquierda Unida 2011 (Figure 2 and the
# example usage) and the six manifestos of Figure A.1.
#
#   sh examples/paper/run.sh PDF_DIR [model] [ollama-host] [doc_id ...]
#
# PDF_DIR holds the original PDFs, named as in manifestos.csv (for example
# 33220_2011.pdf); see README.md for how to obtain them. Without doc ids, only
# IU_2011 is processed.
set -eu

here=examples/paper
pdfs=${1:?usage: run.sh PDF_DIR [model] [ollama-host] [doc_id ...]}
model=${2:-qwen3:14b}
host=${3:-http://localhost:11434}
shift $(( $# < 3 ? $# : 3 ))
[ "$#" -gt 0 ] || set -- IU_2011

# Every argument is quoted, so folders with spaces work.
pt() {
    uv run pledgetrace --pdfs "$pdfs" --work "$here/runs" \
        --corpus "$here/corpus.csv" "$@"
}

# Check the PDFs against the copies used in the paper and print their file stems,
# which are the document ids pledgetrace uses for unannotated PDFs.
stems=$(python3 - "$pdfs" "$here/manifestos.csv" "$@" <<'PY'
import csv, hashlib, sys
from pathlib import Path
folder, table, wanted = Path(sys.argv[1]), sys.argv[2], set(sys.argv[3:])
rows = [r for r in csv.DictReader(open(table, encoding="utf-8")) if r["doc_id"] in wanted]
unknown = wanted - {r["doc_id"] for r in rows}
if unknown:
    sys.exit(f"Unknown document id(s): {', '.join(sorted(unknown))}")
for row in rows:
    path = folder / row["pdf"]
    if not path.exists():
        sys.exit(f"Missing {path} ({row['party']} {row['election']}, "
                 f"Manifesto Project {row['manifesto_project_key']})")
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        print(f"Warning: {path} differs from the copy used in the paper.", file=sys.stderr)
print(" ".join(row["pdf"][:-4] for row in rows))
PY
)

# shellcheck disable=SC2086  # file stems contain no spaces
pt prepare --only $stems
pt pair
# shellcheck disable=SC2086
pt extract --doc $stems --run example --backend ollama --model "$model" --host "$host"
echo "Proposals: $here/runs/<pdf-name>/runs/example/proposals.csv"
