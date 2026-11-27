# Examples from the PledgeTrace paper

The paper illustrates PledgeTrace with the Izquierda Unida manifesto for the 2011 Spanish
general election (Figure 2 and the example usage) and six further manifestos with
different layouts and languages (Figure A.1). This folder lists them and contains a
script to process them. The manual annotations used to evaluate them will be released
separately.

| Document | Manifesto Project key | Used in |
|---|---|---|
| Izquierda Unida, 2011 | `33220_201111` | Figure 2, example usage |
| VOX, 2023 | `33710_202307` | Figure A.1 (a) |
| Partido Popular, 2015 | `33610_201512` | Figure A.1 (b) |
| Unidos Podemos, 2016 | `33210_201606` | Figure A.1 (c) |
| PSOE, November 2019 | `33320_201911` | Figure A.1 (d) |
| En Comú Podem, 2023 | `33096_202307` | Figure A.1 (e) |
| BNG, November 2019 | `33908_201911` | Figure A.1 (f) |

`manifestos.csv` lists, for each document, the file name expected by the script and the
SHA-256 of the PDF used in the paper.

## Getting the PDFs

The PDFs are not included in this repository: they belong to the parties and are
distributed for research by the [Manifesto Project](https://manifesto-project.wzb.eu).

1. Create a free account at <https://manifesto-project.wzb.eu>.
2. Download the original document of each manifesto in the table above. The key
   combines the party code and the election date (`33220_201111` is Izquierda Unida,
   November 2011). R users can open them with `manifestoR::mp_view_originals()`.
3. Save the files in one folder, renamed as in the `pdf` column of `manifestos.csv`
   (for example `33220_2011.pdf`).

`run.sh` compares each file with the SHA-256 of the copy used in the paper and warns if
they differ, so you know whether your results are directly comparable.

## Running

From the repository root, with Ollama running and the model pulled:

```sh
sh examples/paper/run.sh PDF_DIR                                  # IU 2011, qwen3:14b
sh examples/paper/run.sh PDF_DIR MODEL http://SERVER:11434        # another model or server
sh examples/paper/run.sh PDF_DIR MODEL http://localhost:11434 IU_2011 VOX_2023 BNG_201911
```

The paper's figures use `VladimirGav/qwen3.8-27B-14GB-IQ4:latest`. Outputs are written
to `examples/paper/runs/<pdf-name>/`: the reconstructed units in `lin/` and, for the run
`example`, the proposals (`proposals.csv`) and the prompts and raw answers (`chunks/`).
