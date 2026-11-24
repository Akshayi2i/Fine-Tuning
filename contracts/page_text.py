"""How many table rows a page's MinerU text holds (review A1: moved from ``data_pipeline.ocr.run_mineru``,
which re-exports it). The OCR stage records it per page; serving sizes a loss run's windows by it."""

from __future__ import annotations


def count_table_rows(markdown: str) -> int:
    """Count table body rows — markdown pipe tables and MinerU's HTML tables.

    Consumed by the list-completeness cross-check (IMPL-09): if the model extracts
    six claims from a page MinerU saw eight rows on, that is a recall failure the
    per-field confidence cannot see, because the missing rows generate no tokens.

    MinerU writes tables as HTML. Counting pipe tables alone read every MinerU
    page as having no rows, and the row-completeness signal never fired.
    """
    return _count_pipe_rows(markdown) + _count_html_rows(markdown)

def _count_html_rows(text: str) -> int:
    """Body rows of every HTML table, nested tables included, each counted once.

    A row is a header only when it has header cells and no data cells: a schedule
    whose every row starts with ``<th>Vehicle 1</th>`` and continues in ``<td>``
    is all body rows. A table with no header row treats its first row as the
    header, as a pipe table does. Parsed, not matched by regex: a non-greedy
    ``<table>.*?</table>`` stopped at a nested table's end and cut the outer one short.
    """
    from html.parser import HTMLParser

    class _Rows(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.stack: list[list[tuple[bool, bool]]] = []   # per open table: (has_th, has_td) per row
            self.row: list[list[bool]] = []                   # per open table: current row flags
            self.body = 0

        def handle_starttag(self, tag, attrs):
            if tag == "table":
                self.stack.append([])
                self.row.append([False, False])
            elif self.stack and tag == "tr":
                self.row[-1] = [False, False]
            elif self.stack and tag in ("th", "td"):
                self.row[-1][0 if tag == "th" else 1] = True

        def handle_endtag(self, tag):
            if not self.stack:
                return
            if tag == "tr":
                self.stack[-1].append(tuple(self.row[-1]))
            elif tag == "table":
                rows = self.stack.pop()
                self.row.pop()
                headers = sum(1 for has_th, has_td in rows if has_th and not has_td)
                self.body += max(0, len(rows) - (headers or (1 if rows else 0)))

    parser = _Rows()
    parser.feed(text)
    parser.close()
    return parser.body

def _count_pipe_rows(markdown: str) -> int:
    rows = 0
    tables = 0
    in_table = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            in_table = False          # a non-table line ends the current table
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells if c):   # separator row
            continue
        if not in_table:
            # The first row of a run is that table's header.
            in_table = True
            tables += 1
        rows += 1
    # One header per table, not per page. Discounting a single header for the
    # whole page over-reported a two-table page by one row, which raises a false
    # row-completeness flag — and active learning treats that flag as an
    # unconditional override to full manual review.
    return max(0, rows - tables)


#: Prefixed to each page's OCR text. Deliberately visible text rather than an
#: HTML comment: models attend to visible tokens far more reliably, and this
#: marker has a second job — on a window of a longer document it tells the
#: model it is looking at a fragment of it, so the fields that are absent are
#: absent because they live on pages it was not shown. Angle brackets are not
#: Markdown syntax, so the marker cannot be mistaken for document content the
#: way a `#` heading or a `---` rule would be.
PAGE_MARKER = "<page {page} of {total}>"

#: Stands in for a page whose OCR recovered no text, in training AND serving.
#: Serving substituted it and training sent the marker alone, so the model
#: trained on one prompt for a blank page and was served another. An empty
#: string is not an option at serving either: every page is read, a poorly
#: scanned one included, and a request that refused on it would abort exactly
#: the messy documents.
EMPTY_PAGE_TEXT = "(no OCR text recovered for this page — read it from the image)"
