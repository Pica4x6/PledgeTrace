"""Offline tests for the batch pipeline (no model, no corpus files)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pymupdf

from pledgetrace import cli, evaluate, infer, linearize
from pledgetrace.segment import sentences, strip_marker
from pledgetrace.textnorm import norm


class DatasetPathTests(unittest.TestCase):
    def test_commands_default_to_shared_external_corpus(self):
        commands = [
            ["prepare"],
            ["pair"],
            ["ceiling"],
            ["extract", "--run", "example"],
            ["score", "--run", "example"],
        ]
        for command in commands:
            with self.subTest(command=command):
                args = cli.build().parse_args(command)
                self.assertEqual(
                    args.originals, Path("data/pdfs")
                )
                self.assertEqual(args.parsed, Path("data/annotations"))
                self.assertEqual(args.work, Path("runs"))
                self.assertEqual(args.corpus, Path("corpus.csv"))

    def test_custom_dataset_paths_remain_supported(self):
        args = cli.build().parse_args(
            [
                "--originals",
                "custom/pdfs",
                "--parsed",
                "custom/references",
                "score",
                "--run",
                "example",
            ]
        )
        self.assertEqual(args.originals, Path("custom/pdfs"))
        self.assertEqual(args.parsed, Path("custom/references"))


def unit(uid, text, kind="text", page=1, size=10.0, bold=False):
    return {
        "id": uid,
        "page": page,
        "page_end": page,
        "col": 0,
        "size": size,
        "bold": bold,
        "bbox": [0, 0, 1, 1],
        "lines": 1,
        "text": text,
        "kind": kind,
    }


UNITS = [
    unit(1, "EMPLEO", "heading", size=18.0),
    unit(
        2,
        "La situación del empleo juvenil en España es grave. Aprobaremos una ley de empleo juvenil antes de 2025.",
    ),
    unit(3, "• Subiremos el salario mínimo a 1.200 euros.", "item"),
    unit(
        4,
        "12. Crearemos una banca pública a partir de las entidades nacionalizadas.",
        "item",
    ),
    unit(5, "Pie de imprenta"),
]
CHUNK = {"id": "c0001", "first": 1, "last": 5, "context": []}
META = {
    "pdf": "x.pdf",
    "pages": 1,
    "pdf_sha256": "0" * 64,
    "version": linearize.VERSION,
}


def setUpModule():
    infer.PARTS = "prose"  # most tests exercise sentence parts


def tearDownModule():
    infer.PARTS = "none"


class SegmentTests(unittest.TestCase):
    def test_sentences_skip_abbreviations_and_numbering(self):
        text = "1. Reformaremos el art. 135 de la Constitución sin demora. Derogaremos la ley."
        parts = [text[a:b].strip() for a, b in sentences(text, minimum=10)]
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[1].startswith("Derogaremos"))

    def test_strip_marker(self):
        self.assertEqual(strip_marker("•  Texto"), (3, ""))
        offset, number = strip_marker("104.\t Implementaremos")
        self.assertEqual(
            ("104.\t Implementaremos"[offset:], number), ("Implementaremos", "104")
        )
        self.assertEqual(strip_marker("Sin marcador"), (0, ""))
        self.assertEqual(strip_marker("\uf0a7 Texto"), (2, ""))  # Wingdings bullet
        # a quantity with a thousands separator is text, not a proposal number
        self.assertEqual(strip_marker("200.000 apartamentos al término"), (0, ""))
        self.assertEqual(strip_marker("3.1 Título del apartado"), (4, "3.1"))
        self.assertIsNone(linearize.MARKER.match("25.000 habitantes de la isla"))
        # a bullet glyph set tight against the word still opens a list item
        self.assertIsNotNone(linearize.MARKER.match("\uf0b7Revalorizar as pensións"))
        self.assertIsNotNone(linearize.MARKER.match("-\xad\u2010 A corto plazo"))
        self.assertEqual(strip_marker("-\xad\u2010 A corto plazo")[0], 4)

    def test_norm_ignores_case_accents_and_punctuation(self):
        self.assertEqual(norm("Liberali-\nzación, ¡YA!"), "liberalizacionya")


class InferTests(unittest.TestCase):
    def test_parse_and_rows_rebuild_text_from_units(self):
        labels = infer.parse('{"h": [[1, 1]], "p": ["2b", "3", "4"]}', UNITS, CHUNK)
        rows = infer.build_rows(UNITS, [labels], META)
        self.assertEqual([row["seccion1"] for row in rows], ["EMPLEO"] * 3)
        self.assertEqual(
            rows[0]["propuesta_texto"],
            "Aprobaremos una ley de empleo juvenil antes de 2025.",
        )
        self.assertEqual(
            rows[1]["propuesta_texto"], "Subiremos el salario mínimo a 1.200 euros."
        )
        self.assertEqual(rows[2]["propuesta_id"], "12")
        for row in rows:  # provenance points at the exact source characters
            source = UNITS[row["unit_from"] - 1]["text"][
                row["char_from"] : row["char_to"]
            ]
            self.assertEqual(source, row["propuesta_texto"])

    def test_parse_rejects_bad_answers(self):
        for raw in (
            "no json",
            '{"p": ["9"]}',
            '{"p": ["2", "2b"]}',
            '{"p": ["3-2"]}',
            '{"p": ["2c"]}',
            '{"h": [1], "p": ["1"]}',
            '{"h": [1], "p": ["1-2"]}',
        ):
            with self.assertRaises(ValueError, msg=raw):
                infer.parse(raw, UNITS, CHUNK)

    def test_ranges_cannot_swallow_lists_or_recurring_headings(self):
        long = "Aprobaremos una ley integral que regule los precios del alquiler en zonas tensionadas."
        units = [unit(i, f"{i}. Apartado", "heading", size=14.0) for i in (1, 2, 3)]
        units += [unit(4, "• " + long, "item"), unit(5, "• " + long, "item")]
        units += [
            unit(6, "• Reformas laborales", "item"),
            unit(7, "• Ley Mordaza", "item"),
        ]
        chunk = {"id": "c0001", "first": 1, "last": 7, "context": []}
        for raw in ('{"p": ["3-4"]}', '{"p": ["4-5"]}'):
            with self.assertRaises(ValueError, msg=raw):
                infer.parse(raw, units, chunk)
        self.assertEqual(
            len(infer.parse('{"p": ["4", "5", "6-7"]}', units, chunk)["p"]), 3
        )
        fixed = infer.parse(
            '{"h": [3], "p": ["3-5", "6-7"]}', units, chunk, repair=True
        )
        self.assertEqual([p["unit_from"] for p in fixed["p"]], [4, 5, 6])
        self.assertEqual(fixed["repaired"], 1)
        loose = '{"p": ["4a-4c", "9", "zz", "4", "5c"]}'  # clamp, drop, drop, overlap, whole
        fixed = infer.parse(loose, units, chunk, repair=True)
        self.assertEqual([p["unit_from"] for p in fixed["p"]], [4, 5])
        self.assertEqual(len(infer.parse('{"p": ["4a-4c"]}', units, chunk)["p"]), 1)

    def test_whole_unit_mode_has_no_parts(self):
        infer.PARTS = "none"
        try:
            self.assertNotIn("{a}", infer.render(UNITS[1]))
            self.assertNotIn("15a", infer.system_prompt())
            labels = infer.parse(
                '{"p": ["2b", "3"]}', UNITS, CHUNK
            )  # stray letter ignored
            rows = infer.build_rows(UNITS, [labels], META)
            self.assertEqual(rows[0]["propuesta_texto"], UNITS[1]["text"])
        finally:
            infer.PARTS = "prose"

    def test_lead_in_lines_are_not_exported(self):
        units = [
            unit(1, "Para ello proponemos:"),
            unit(2, "Crearemos una banca pública."),
        ]
        chunk = {"id": "c0001", "first": 1, "last": 2, "context": []}
        rows = infer.build_rows(
            units, [infer.parse('{"p": ["1", "2"]}', units, chunk)], META
        )
        self.assertEqual([row["unit_from"] for row in rows], [2])

    def test_summary_block_is_marked_but_body_is_kept(self):
        topics = [f"materia{n:02d} prioridad{n:02d} reforma{n:02d}" for n in range(25)]
        summary = [{"propuesta_texto": f"Nueva {t}."} for t in topics]
        body = [
            {
                "propuesta_texto": f"Aprobaremos la {t} con dotación suficiente y calendario "
                * 3
            }
            for t in topics
        ]
        marked = infer.summary_blocks(summary + body)
        self.assertTrue(all(marked[:25]))
        self.assertFalse(any(marked[25:]))
        self.assertFalse(any(infer.summary_blocks(body)))

    def test_paragraph_long_heading_is_kept_as_proposal(self):
        long = (
            "7. Ejecutar las inversiones comprometidas. "
            + "Reclamaremos el cumplimiento. " * 5
        )
        units = [unit(1, "Autogobierno", "heading", size=14.0), unit(2, long, "item")]
        chunk = {"id": "c0001", "first": 1, "last": 2, "context": []}
        labels = infer.parse('{"h": [1, 2], "p": []}', units, chunk)
        self.assertEqual((labels["h"], labels["promoted"]), ([1], 1))
        self.assertEqual(labels["p"][0]["unit_from"], 2)

    def test_document_numbering_delimits_proposals(self):
        units = []
        for n in range(1, 121):
            units.append(
                unit(
                    0, f"{n}. Impulsar la medida número {n} para todo el país.", "item"
                )
            )
            units.append(
                unit(0, f"Aprobaremos la ley {n} con dotación suficiente y calendario.")
            )
        units.append(unit(0, "Otro capítulo", "heading", size=18.0))
        for index, item in enumerate(units, 1):
            item["id"] = index
        chunk = {"id": "c0001", "first": 1, "last": len(units), "context": []}
        # the model takes only the statement of 1, only the development of 2, splits 3
        labels = infer.parse('{"h": [], "p": ["1", "4", "5", "6"]}', units, chunk)
        rows = infer.build_rows(units, [labels], META)
        spans = [(row["unit_from"], row["unit_to"]) for row in rows]
        self.assertEqual(spans[:3], [(1, 2), (3, 4), (5, 6)])
        self.assertEqual(
            len(rows), 120
        )  # 119 entries of two units + the last statement
        self.assertEqual(rows[0]["propuesta_id"], "1")
        self.assertEqual(
            infer.enumeration(units[:60]), []
        )  # a short list is not trusted

    def test_sub_numbered_items_are_proposals_of_their_own(self):
        units = []
        for n in range(1, 121):
            units.append(
                unit(0, f"{n}. Impulsar la medida número {n} en todo el país.", "item")
            )
            if n == 50:
                units.append(
                    unit(0, "50.1 Ningún centro quedará fuera de la medida.", "item")
                )
                units.append(
                    unit(
                        0, "50.2 Se creará un fondo específico para aplicarla.", "item"
                    )
                )
        for index, item in enumerate(units, 1):
            item["id"] = index
        chunk = {"id": "c0001", "first": 1, "last": len(units), "context": []}
        labels = infer.parse('{"h": [], "p": ["50-52"]}', units, chunk)
        rows = infer.build_rows(units, [labels], META)
        ids = [
            row["propuesta_id"] for row in rows if row["propuesta_id"].startswith("50")
        ]
        self.assertEqual(ids, ["50", "50.1", "50.2"])

    def test_a_quantity_at_the_start_stays_in_the_text(self):
        units = [unit(1, "50 razones para luchar por Canarias y su futuro común.")]
        chunk = {"id": "c0001", "first": 1, "last": 1, "context": []}
        rows = infer.build_rows(
            units, [infer.parse('{"p": ["1"]}', units, chunk)], META
        )
        self.assertEqual(rows[0]["propuesta_id"], "")
        self.assertTrue(rows[0]["propuesta_texto"].startswith("50 razones"))

    def test_index_without_dot_leaders_is_not_exported(self):
        titles = [f"Capítulo sobre la materia {chr(65 + n)}" for n in range(10)]
        units = [unit(0, f"{chr(97 + n)}) {t}", "item") for n, t in enumerate(titles)]
        for title in titles:
            units.append(unit(0, title, "heading", size=16.0))
            units.append(
                unit(0, f"Aprobaremos una ley para {title.lower()} con dotación.")
            )
        for position, item in enumerate(units, 1):
            item["id"] = position
        self.assertEqual(infer.index_entries(units), set(range(1, 11)))
        chunk = {"id": "c0001", "first": 1, "last": len(units), "context": []}
        labels = infer.parse('{"p": ["1", "2", "3", "12", "14"]}', units, chunk)
        rows = infer.build_rows(units, [labels], META)
        self.assertEqual([row["unit_from"] for row in rows], [12, 14])

    def test_preamble_of_numbered_sections(self):
        units, number = [], 1
        for chapter in range(3):
            units.append(unit(0, f"Capítulo {chapter}", "heading", size=18.0))
            units.append(unit(0, "Hemos hecho mucho en esta legislatura por el país."))
            for _ in range(40):
                units.append(
                    unit(0, f"{number}. Aprobaremos la medida {number}.", "item")
                )
                number += 1
        for position, item in enumerate(units, 1):
            item["id"] = position
        preamble = infer.preamble_units(units)
        self.assertEqual(preamble, {2, 44, 86})
        self.assertEqual(infer.preamble_units(units[:30]), set())  # no long numbering

    def test_ranges_of_separate_paragraphs_are_split(self):
        long = " con dotación suficiente y un calendario de aplicación en toda España."
        units = [
            unit(1, "Aprobaremos una ley de vivienda" + long),
            unit(2, "Crearemos un parque público de alquiler" + long),
            unit(3, "Reformaremos la ley hipotecaria" + long),
            unit(4, "y además revisaremos los desahucios" + long),
        ]
        chunk = {"id": "c0001", "first": 1, "last": 4, "context": []}
        labels = [infer.parse('{"p": ["1-2", "3-4"]}', units, chunk)]
        self.assertEqual(infer.split_merged_ranges(units, labels), 1)
        spans = [(p["unit_from"], p["unit_to"]) for p in labels[0]["p"]]
        self.assertEqual(spans, [(1, 1), (2, 2), (3, 4)])

    def test_chunks_cover_every_unit_once(self):
        units = [
            unit(i, "x" * 400, "heading" if i % 7 == 0 else "text")
            for i in range(1, 60)
        ]
        chunks = infer.plan_chunks(units, chars=3000)
        covered = [i for c in chunks for i in range(c["first"], c["last"] + 1)]
        self.assertEqual(covered, list(range(1, 60)))
        self.assertEqual(chunks[1]["first"], 7)  # cut falls before a heading candidate
        self.assertIn("[7 T", infer.user_prompt(units, chunks[1], META))

    def test_heading_levels_follow_style_not_the_model(self):
        units = [
            unit(1, "PROGRAMA", "heading", size=30.0),
            unit(6, "Capítulo", "heading", size=18.0),
            unit(2, "Apartado", "heading", size=12.0, bold=True),
            unit(3, "Aprobaremos una ley de vivienda que regule los alquileres."),
            unit(4, "Otro capítulo", "heading", size=18.0),
            unit(5, "Crearemos un parque público de cincuenta mil viviendas."),
        ]
        chunk = {"id": "c0001", "first": 1, "last": 6, "context": []}
        units = [units[0], units[1]] + units[2:]
        for index, item in enumerate(units, 1):
            item["id"] = index
        labels = infer.parse('{"h": [[1, 3], 2, 3, 5], "p": ["4", "6"]}', units, chunk)
        rows = infer.build_rows(units, [labels], META)
        self.assertEqual(
            (rows[0]["seccion1"], rows[0]["seccion2"]), ("Capítulo", "Apartado")
        )
        self.assertEqual(
            (rows[1]["seccion1"], rows[1]["seccion2"]), ("Otro capítulo", "")
        )


class EvaluateTests(unittest.TestCase):
    def test_score_and_ceiling(self):
        reference = [
            {
                "seccion1": "Empleo",
                "propuesta_texto": "Aprobaremos una ley de empleo juvenil antes de 2025.",
            },
            {
                "seccion1": "Empleo",
                "propuesta_texto": "Subiremos el salario minimo a 1.200 euros.",
            },
            {
                "seccion1": "Empleo",
                "propuesta_texto": "Texto que no existe en el documento original.",
            },
        ]
        labels = infer.parse('{"h": [[1, 1]], "p": ["2b", "3", "5"]}', UNITS, CHUNK)
        result = evaluate.score(
            UNITS, reference, infer.build_rows(UNITS, [labels], META)
        )
        self.assertEqual(result["reference_located"], 2)
        self.assertEqual(result["strict_matches"], 2)
        self.assertEqual(result["unsupported"], 1)
        self.assertEqual(result["seccion1_agreement"], 1.0)
        ceiling = evaluate.ceiling(UNITS, reference)
        self.assertEqual((ceiling["unit_share"], ceiling["sentence_share"]), (0.5, 1.0))


class LinearizeTests(unittest.TestCase):
    def _line(self, x0, x1, text, y0=100.0, size=11.0, symbol=False):
        return {
            "page": 1,
            "text": text,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y0 + size,
            "size": size,
            "bold": False,
            "rotated": False,
            "bullet": False,
            "lead_bold": False,
            "symbol": symbol,
        }

    def test_overprinted_copies_are_dropped(self):
        lines = [
            self._line(100, 300, "RNE e TVE, especialmente daqueles"),
            self._line(250, 300, "daqueles"),
            self._line(90, 98, "2", size=26.0),
            self._line(90.5, 98, "2", size=26.0),
        ]
        kept = linearize._drop_overprints(lines)
        self.assertEqual(
            [w["text"] for w in kept], ["RNE e TVE, especialmente daqueles", "2"]
        )

    def test_words_of_a_justified_row_are_joined_but_columns_are_not(self):
        lines = [
            self._line(100, 150, "especialmente"),
            self._line(161, 200, "daqueles"),
            self._line(211, 260, "programas"),
            self._line(320, 400, "outra columna"),
        ]
        joined = linearize._join_words(lines)
        self.assertEqual(
            [w["text"] for w in joined],
            ["especialmente daqueles programas", "outra columna"],
        )

    def test_large_list_number_is_not_appended_to_text_in_another_column(self):
        # PP 2023 p. 15: a 22 pt pull quote on the left, «15» at 39 pt opening
        # the item in the right column; the number must not end the quote.
        lines = [
            self._line(90, 243, "competitiva. Para", y0=380, size=22.0),
            self._line(283, 319, "15", y0=366, size=39.0),
        ]
        joined = linearize._join_words(lines)
        self.assertEqual(
            sorted(w["text"] for w in joined), ["15", "competitiva. Para"]
        )

    def test_list_number_still_joins_the_text_it_opens(self):
        lines = [
            self._line(60, 72, "3", size=16.0),
            self._line(90, 400, "Implantaremos un plan", size=11.0),
        ]
        joined = linearize._join_words(lines)
        self.assertEqual([w["text"] for w in joined], ["3 Implantaremos un plan"])

    def test_margin_number_beside_a_later_line_of_its_entry_is_not_kept(self):
        # PP 2015: «50» is centred on its entry, beside the second line
        lines = [
            self._line(112, 570, "Impulsaremos que un mínimo de la emisión", y0=288, size=9.0),
            self._line(85, 101, "50", y0=300.3, size=9.0),
            self._line(112, 570, "produzca en versión original en la televisión", y0=301, size=9.0),
        ]
        self.assertFalse(linearize._entry_number(lines[1], lines))

    def test_number_after_an_entry_without_final_stop_is_kept(self):
        # Compromís 2016: entry 80 ends without a full stop; «81.» opens the next
        lines = [
            self._line(45, 62, "80.", y0=416, size=12.0),
            self._line(79, 500, "Integrarem i coordinarem els recursos", y0=416, size=12.0),
            self._line(79, 390, "per a la presa de decisions responsables", y0=431, size=12.0),
            self._line(45, 62, "81.", y0=446, size=12.0),
            self._line(79, 500, "Treballarem per a incorporar la salut", y0=446, size=12.0),
        ]
        self.assertTrue(linearize._entry_number(lines[3], lines))

    def test_number_in_its_own_narrow_column_does_not_cross_the_gutter(self):
        # Compromís 2019-04: «4.» stands between two text columns
        lines = []
        for row in range(14):
            y = 100 + 13 * row
            lines.append(self._line(57, 390, f"izquierda {row} con texto largo", y0=y, size=9.0))
            lines.append(self._line(506, 780, f"derecha {row} con texto largo", y0=y, size=8.5))
        number = self._line(478, 492, "4.", y0=113, size=14.0)
        joined = linearize._join_words(lines + [number])
        opened = [w for w in joined if w["text"].startswith("4. ")]
        self.assertEqual(len(opened), 1)
        ordered = linearize._order(joined, 842.0)
        texts = [w["text"] for w in ordered]
        self.assertLess(texts.index("izquierda 13 con texto largo"), texts.index(opened[0]["text"]))

    def test_running_header_with_changing_chapter_is_furniture(self):
        pages = []
        for n, chapter in enumerate(["Salud", "Educación", "Empleo", "Cultura"]):
            header = self._line(600, 780, f"Programa 2019 / {chapter}", y0=33.0)
            body = self._line(
                60, 400, f"Texto del capítulo {chapter} con medidas.", y0=200.0
            )
            pages.append((595.0, [header, body]))
        often, _, _ = linearize._furniture(pages)
        self.assertIn(linearize._key(pages[0][1][0]), often)
        self.assertNotIn(linearize._key(pages[0][1][1]), often)

    def test_repeated_first_line_of_a_changing_title_is_kept(self):
        title = self._line(
            60, 400, "ALGUNAS DE LAS ACCIONES MÁS DESTACADAS", y0=53, size=17.0
        )
        rest = self._line(110, 350, "DE VOX SOBRE ESPAÑA RURAL", y0=73, size=17.0)
        header = self._line(60, 200, "PROGRAMA", y0=20, size=17.0)
        below = self._line(60, 200, "ELECTORAL", y0=38, size=17.0)
        body = [
            self._line(60, 500, "Texto", y0=200 + 12 * k, size=9.0) for k in range(9)
        ]
        lines = [title, rest, header, below, *body]
        repeated = {id(title), id(header), id(below)}
        self.assertTrue(
            linearize._title_start(title, lines, 9.0, lambda o: id(o) in repeated)
        )
        self.assertFalse(
            linearize._title_start(header, lines, 9.0, lambda o: id(o) in repeated)
        )

    def test_short_words_are_not_taken_for_bullet_glyphs(self):
        def line(x0, x1, text, symbol=False):
            return {
                "page": 1,
                "text": text,
                "x0": x0,
                "y0": 100.0,
                "x1": x1,
                "y1": 111.0,
                "size": 11.0,
                "bold": False,
                "rotated": False,
                "bullet": False,
                "lead_bold": False,
                "symbol": symbol,
            }

        words = [
            line(100, 160, "impulsando"),
            line(163, 172, "la"),
            line(175, 230, "atracción"),
            line(233, 245, "de"),
        ]
        kept = linearize._attach_glyphs(words)
        self.assertEqual(
            [w["text"] for w in kept], ["impulsando", "la", "atracción", "de"]
        )
        bullet = [line(90, 96, "\uf0a7", symbol=True), line(100, 300, "Primera medida")]
        kept = linearize._attach_glyphs(bullet)
        self.assertEqual([w["text"] for w in kept], ["Primera medida"])
        self.assertTrue(kept[0]["bullet"])

    def test_columns_furniture_and_hyphenation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "doc.pdf"
            document = pymupdf.open()
            for number, word in enumerate(["alfa", "beta", "gamma", "delta"], 1):
                page = document.new_page(width=600, height=800)
                page.insert_text((250, 780), f"Programa electoral {number}", fontsize=8)
                for row in range(8):
                    y = 100 + row * 14
                    left = (
                        f"Primera columna {word} con una pala-"
                        if row == 0
                        else f"bra partida y texto {word} seguido"
                    )
                    page.insert_text((50, y), left, fontsize=10)
                    page.insert_text(
                        (330, y), f"Segunda columna {word} aquí", fontsize=10
                    )
            document.save(path)
            document.close()
            result = linearize.linearize(path)
        texts = [item["text"] for item in result["units"]]
        self.assertFalse(any("Programa electoral" in text for text in texts))
        self.assertTrue(any("palabra partida" in text for text in texts))
        first_page = [item for item in result["units"] if item["page"] == 1]
        self.assertTrue(first_page[0]["text"].startswith("Primera"))
        self.assertTrue(first_page[-1]["text"].startswith("Segunda"))
        self.assertEqual(result["meta"]["pages_with_columns"], [1, 2, 3, 4])

    def test_large_entry_numbers_and_two_line_capital_leads(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "doc.pdf"
            document = pymupdf.open()
            number = 1
            for _ in range(3):
                page = document.new_page(width=600, height=800)
                y = 120
                for _ in range(4):
                    page.insert_text((60, y + 10), str(number), fontsize=36)
                    word = "ABCDEFGHIJKLMNOP"[number] * 3
                    lead = (
                        f"APROBAREMOS LA LEY {word} PARA ORDENAR TODAS LAS MATERIAS DE "
                    )
                    page.insert_text((100, y), lead, fontsize=10, fontname="hebo")
                    page.insert_text(
                        (100, y + 13), f"LA MEDIDA {word}.", fontname="hebo"
                    )
                    page.insert_text(
                        (190, y + 13),
                        "Con dotación suficiente y calendario.",
                        fontsize=10,
                    )
                    y += 120
                    number += 1
            document.save(path)
            document.close()
            units = linearize.linearize(path)["units"]
        entries = [u for u in units if u["text"][:1].isdigit()]
        self.assertEqual(
            [u["text"].split(".")[0] for u in entries], [str(n) for n in range(1, 13)]
        )
        self.assertTrue(
            all("calendario" in u["text"] for u in entries)
        )  # lead + development

    def test_versioned_cache_keeps_older_units(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            old = {
                "meta": {"version": "linearize-0", "pdf_sha256": "x"},
                "units": [unit(1, "a")],
            }
            linearize.save(old, directory)
            linearize.save(
                {
                    "meta": {"version": linearize.VERSION, "pdf_sha256": "x"},
                    "units": [],
                },
                directory / "lin" / linearize.VERSION,
            )
            self.assertEqual(
                linearize.load_version(directory, "linearize-0")["units"][0]["id"], 1
            )
            self.assertEqual(linearize.load_current(directory)["units"], [])


if __name__ == "__main__":
    unittest.main()


class OllamaRetrySamplingTest(unittest.TestCase):
    def test_first_attempt_is_greedy_and_retries_sample_with_a_seed(self):
        from pledgetrace import backends

        sent = []

        def fake_post(url, payload, headers, timeout):
            sent.append(payload["options"])
            return {"message": {"content": "{}"}}

        original = backends._post
        backends._post = fake_post
        try:
            client = backends.Ollama("m")
            client.complete("s", "u")
            client.complete("s", "u", attempt=1)
            client.complete("s", "u", attempt=2)
        finally:
            backends._post = original
        self.assertEqual(sent[0]["temperature"], 0)
        self.assertNotIn("seed", sent[0])
        self.assertEqual((sent[1]["temperature"], sent[1]["seed"]), (backends.RETRY_TEMPERATURE, 1))
        self.assertEqual(sent[2]["seed"], 2)


class CommandLineTest(unittest.TestCase):
    def test_documents_can_be_repeated_or_listed(self):
        args = cli.build().parse_args(["extract", "--run", "r", "--doc", "A", "--doc", "B", "C"])
        self.assertEqual(args.doc, ["A", "B", "C"])

    def test_path_options_have_readable_aliases(self):
        args = cli.build().parse_args(["--pdfs", "p", "--annotations", "a", "pair"])
        self.assertEqual((args.originals, args.parsed), (Path("p"), Path("a")))


class InputRobustnessTest(unittest.TestCase):
    def test_null_heading_list_is_an_invalid_answer_not_a_crash(self):
        units = [{"id": 1, "text": "Propuesta uno.", "kind": "text", "page": 1}]
        chunk = {"id": "c0001", "first": 1, "last": 1}
        with self.assertRaises(ValueError):
            infer.parse('{"h": null, "p": []}', units, chunk)

    def test_prepare_fails_on_a_missing_pdf_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = cli.build().parse_args(
                ["--pdfs", str(Path(tmp) / "missing"), "--work", tmp, "prepare"]
            )
            self.assertEqual(cli.cmd_prepare(args), 2)


class BackendSecurityTest(unittest.TestCase):
    def test_api_key_is_not_sent_over_plain_http_to_another_host(self):
        from pledgetrace import backends

        with self.assertRaises(ValueError):
            backends._check_secure("http://example.org/v1", {"Authorization": "Bearer x"})
        backends._check_secure("http://localhost:8000/v1", {"Authorization": "Bearer x"})
        backends._check_secure("http://example.org/v1", {})

    def test_redirects_are_refused(self):
        import urllib.error
        import urllib.request

        from pledgetrace import backends

        handler = backends._NoRedirect()
        request = urllib.request.Request("https://api.example.org/v1")
        with self.assertRaises(urllib.error.HTTPError):
            handler.redirect_request(request, None, 302, "Found", {}, "http://evil.example/")
