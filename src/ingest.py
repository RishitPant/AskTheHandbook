import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DATA_DIR = Path("data")
DB_DIR = "db"
COLLECTION_NAME = "handbook_docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
USE_CHROMA_CLOUD = bool(CHROMA_API_KEY)


def get_chroma_client():
    """Return a chromadb client — CloudClient if Chroma Cloud env vars are
    set, otherwise a local PersistentClient writing to DB_DIR."""
    import chromadb

    if USE_CHROMA_CLOUD:
        return chromadb.CloudClient(
            api_key=CHROMA_API_KEY,
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
        )
    return chromadb.PersistentClient(path=DB_DIR)

HEADERS_TO_SPLIT = [
    ("#",   "h1"),
    ("##",  "h2"),
    ("###", "h3"),
]

_TABLE_LINK_MAP = {
    r'<u>\s*Table\s*2\s*</u>': (
        "Table 2 (NPTEL Dep/Free Electives — "
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSJXV0JECyoQvgWvBlVxO13G0KRm5a1qNCRBa7rAw8GDY4e0cfm1KiVCwlgs_ed80ObtzQ1rfx_JWIR/pub?gid=399341609&single=true)"
    ),
    r'<u>\s*NPTEL-Table\s*3\s*</u>': (
        "Table 3 NPTEL HS/MG Electives — "
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSJXV0JECyoQvgWvBlVxO13G0KRm5a1qNCRBa7rAw8GDY4e0cfm1KiVCwlgs_ed80ObtzQ1rfx_JWIR/pub?gid=1418834182&single=true)"
    ),
}

def expand_table_links(text: str) -> str:
    """Replace bare <u>Table N</u> anchors with their full resolved URLs."""
    import re
    for pattern, replacement in _TABLE_LINK_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def _extract_table_rows(table_html: str) -> list[list[str]]:
    """Extract all cell values from an HTML table as a list of rows (no rowspan handling)."""
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)
    result = []
    for row in rows:
        cells = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', row, re.DOTALL | re.IGNORECASE)
        cleaned = [re.sub(r'<[^>]+>', '', cell).strip() for cell in cells]
        if any(cleaned):
            result.append(cleaned)
    return result


def _parse_cell(attrs: str, content: str) -> dict:
    """Parse a single table cell including rowspan/colspan."""
    text = re.sub(r'<br\s*/?>', ' ', content, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text).strip()
    rowspan = int(re.search(r'rowspan=["\']?(\d+)', attrs, re.I).group(1)) if re.search(r'rowspan', attrs, re.I) else 1
    colspan = int(re.search(r'colspan=["\']?(\d+)', attrs, re.I).group(1)) if re.search(r'colspan', attrs, re.I) else 1
    return {'text': text, 'rowspan': rowspan, 'colspan': colspan}


def _extract_cells_raw(table_html: str) -> list[list[dict]]:
    """Extract rows as dicts with text/rowspan/colspan preserved."""
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)
    return [
        [_parse_cell(a, c) for a, c in re.findall(r'<t[hd]([^>]*)>(.*?)</t[hd]>', row, re.DOTALL | re.IGNORECASE)]
        for row in rows
    ]


def _expand_rowspans(raw_rows: list, n_cols: int) -> list[list[str]]:
    """Expand rowspan/colspan into a full 2D grid of strings."""
    grid = []
    carry: dict = {}
    for row in raw_rows:
        expanded: list = []
        col = 0
        cell_iter = iter(row)
        while col < n_cols:
            if col in carry:
                text, rem = carry[col]
                expanded.append(text)
                if rem > 1:
                    carry[col] = (text, rem - 1)
                else:
                    del carry[col]
                col += 1
            else:
                try:
                    cell = next(cell_iter)
                except StopIteration:
                    expanded.append('')
                    col += 1
                    continue
                for c in range(cell['colspan']):
                    expanded.append(cell['text'])
                    if cell['rowspan'] > 1:
                        carry[col + c] = (cell['text'], cell['rowspan'] - 1)
                col += cell['colspan']
        grid.append(expanded[:n_cols])
    return grid


# --- OPPE Schedule table conversion ---

_OPPE_SCHEDULE_HEADING = re.compile(r'OPPE\s+SCHEDULE', re.IGNORECASE)
_OPPE_CONTINUATION = re.compile(r'OPPE\s*2\s*\(Day\s*[34]\)', re.IGNORECASE)


def _is_oppe_header_row(row_cells: list) -> bool:
    texts = {c['text'] for c in row_cells}
    return bool({'Exam', 'Python', 'Timing'} & texts)


def _is_date_row(grid_row: list) -> bool:
    non_empty = [c for c in grid_row if c.strip()]
    unique = set(non_empty)
    return len(unique) == 1 and bool(re.search(r'\b20\d\d\b', list(unique)[0]))


def _oppe_table_to_prose(table_html: str, inherited_date: str = '') -> tuple:
    """
    Convert an OPPE schedule table (or fragment) into prose lines like:
      "OPPE1 (Day 1) on Saturday, August 1, 2026, 2.30 PM to 4.30 PM: Python"

    Returns (lines, last_date_seen) so split fragments can pass the date forward.
    """
    raw_rows = _extract_cells_raw(table_html)
    if not raw_rows:
        return [], inherited_date

    header_indices = [i for i, r in enumerate(raw_rows) if _is_oppe_header_row(r)]
    if not header_indices:
        return [], inherited_date

    first_hdr = raw_rows[header_indices[0]]
    headers: list = []
    for c in first_hdr:
        for _ in range(c['colspan']):
            headers.append(c['text'])
    n_cols = len(headers)

    col_exam   = next((i for i, h in enumerate(headers) if h == 'Exam'), 0)
    col_timing = next((i for i, h in enumerate(headers) if 'iming' in h), 1)
    subject_cols = [(i, h) for i, h in enumerate(headers) if i not in (col_exam, col_timing) and h.strip()]

    data_rows = [row for i, row in enumerate(raw_rows) if i not in set(header_indices)]
    grid = _expand_rowspans(data_rows, n_cols)

    current_date = inherited_date
    current_exam = ''
    lines: list = []

    for row in grid:
        if not row or len(row) < 2:
            continue
        if _is_date_row(row):
            current_date = next(c for c in row if c.strip())
            continue
        exam   = row[col_exam].strip() if col_exam < len(row) else ''
        timing = row[col_timing].strip() if col_timing < len(row) else ''
        if exam and exam != 'Exam':
            current_exam = exam
        subjects = [
            h for i, h in subject_cols
            if i < len(row) and row[i].strip() and not re.search(r'\b20\d\d\b', row[i])
        ]
        if subjects and timing and current_exam:
            lines.append(f"{current_exam} on {current_date}, {timing}: {', '.join(subjects)}")

    return lines, current_date


def _electives_table_to_prose(table_html: str) -> str:
    """
    Convert the Department Core/Elective Courses table into term-grouped
    prose sentences instead of a pipe table.

    Each term gets one sentence listing every course offered that term,
    e.g.:
      Courses offered in May 2026: Software Engineering (BSCS3001, Core_BP),
      Deep Learning (BSCS3004, Core_BD), ...

    This keeps the full content in a small, naturally retrievable chunk.
    """
    rows = _extract_table_rows(table_html)

    # Find the header row that contains "Course ID"
    header_idx = next(
        (i for i, r in enumerate(rows) if any("Course ID" in c for c in r)), None
    )
    if header_idx is None:
        # Fallback: convert normally if structure is unexpected
        return _table_to_markdown(table_html)

    header = rows[header_idx]
    # Locate column indices dynamically
    def col(name):
        for i, h in enumerate(header):
            if name.lower() in h.lower():
                return i
        return None

    idx_id    = col("Course ID")
    idx_name  = col("Course Name")
    idx_type  = col("Course Type")
    idx_level = col("Course Level")
    idx_may   = col("May 2026")
    idx_sep   = col("Sep 2026")
    idx_jan   = col("Jan 2027")

    if any(i is None for i in [idx_id, idx_name, idx_may, idx_sep, idx_jan]):
        return _table_to_markdown(table_html)

    data_rows = rows[header_idx + 1:]

    def collect_term(term_idx):
        courses = []
        for r in data_rows:
            if len(r) > term_idx and r[term_idx].strip().upper() == 'Y':
                cid   = r[idx_id].strip()   if idx_id   < len(r) else ''
                cname = r[idx_name].strip()  if idx_name < len(r) else ''
                ctype = r[idx_type].strip()  if idx_type is not None and idx_type < len(r) else ''
                clvl  = r[idx_level].strip() if idx_level is not None and idx_level < len(r) else ''
                parts = f"{cname} ({cid}"
                if ctype:
                    parts += f", {ctype}"
                if clvl:
                    parts += f", {clvl}"
                parts += ")"
                courses.append(parts)
        return courses

    lines = []
    for term_label, term_idx in [("May 2026", idx_may), ("Sep 2026", idx_sep), ("Jan 2027", idx_jan)]:
        courses = collect_term(term_idx)
        if courses:
            lines.append(f"Courses offered in {term_label}: {', '.join(courses)}.")

    return "\n\n" + "\n\n".join(lines) + "\n\n"


def _table_to_markdown(table_html: str) -> str:
    """Default HTML table → Markdown pipe table (used for all non-elective tables)."""
    rows = _extract_table_rows(table_html)
    if not rows:
        return table_html
    md_rows = ['| ' + ' | '.join(r) + ' |' for r in rows]
    separator = '| ' + ' | '.join(['---'] * len(rows[0])) + ' |'
    md_rows.insert(1, separator)
    return '\n' + '\n'.join(md_rows) + '\n'


# Heading text that immediately precedes the electives table in the source doc
_ELECTIVES_TABLE_HEADING = re.compile(
    r'Table\s+1\s*:\s*Department\s+Core/Elective\s+Courses',
    re.IGNORECASE,
)

# Matches a continuation fragment: <table><tbody> with no <thead>, first cell is a course ID
_ELECTIVES_CONTINUATION = re.compile(
    r'^<table[^>]*>\s*<tbody>\s*<tr>\s*<td>\s*BS[A-Z0-9]+\s*</td>',
    re.IGNORECASE,
)

def stitch_electives_table(text: str) -> str:
    """
    The Google Docs export inserts page-break boilerplate mid-table, producing
    two separate <table> blocks for Table 1. Detect the continuation fragment
    (no <thead>, first cell is a course ID like BSMA3014) and merge its <tbody>
    rows back into the preceding table before any conversion happens.
    """
    table_pattern = re.compile(r'(<table[^>]*>.*?</table>)', re.DOTALL | re.IGNORECASE)
    parts = table_pattern.split(text)
    # parts alternates: [text, table, text, table, ...]
    i = 0
    while i < len(parts):
        if i >= 2 and table_pattern.match(parts[i]) and _ELECTIVES_CONTINUATION.match(parts[i].strip()):
            # Extract just the <tr>...</tr> rows from the continuation tbody
            extra_rows = re.findall(r'(<tr[^>]*>.*?</tr>)', parts[i], re.DOTALL | re.IGNORECASE)
            if extra_rows:
                # Inject them before </tbody></table> of the preceding table (parts[i-2])
                parts[i - 2] = re.sub(
                    r'(</tbody>\s*</table>)\s*$',
                    '\n'.join(extra_rows) + r'\n</tbody>\n</table>',
                    parts[i - 2],
                    flags=re.DOTALL | re.IGNORECASE,
                )
                parts[i] = ''       # remove the now-merged fragment
                parts[i - 1] = ''   # remove the noise between them
        i += 1
    return ''.join(parts)


def html_tables_to_markdown(text: str) -> str:
    """
    1. Stitch the page-break-split electives table back into one block.
    2. Convert the stitched electives table → term-grouped prose sentences.
    3. Convert OPPE schedule tables (including page-break fragments) → prose lines,
       carrying the last-seen date from fragment 1 into fragment 2.
    4. Convert all other tables → standard Markdown pipe table.
    """
    text = stitch_electives_table(text)

    table_pattern = re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE)
    oppe_date_carry = ['']  # mutable so nested func can update it

    def convert_table(match):
        table_html = match.group(0)
        start = match.start()
        preceding = text[max(0, start - 300): start]

        if _ELECTIVES_TABLE_HEADING.search(preceding):
            return _electives_table_to_prose(table_html)

        if _OPPE_SCHEDULE_HEADING.search(preceding) or _OPPE_SCHEDULE_HEADING.search(table_html[:200]) or _OPPE_CONTINUATION.search(table_html):
            lines, last_date = _oppe_table_to_prose(table_html, inherited_date=oppe_date_carry[0])
            oppe_date_carry[0] = last_date
            if lines:
                return "\n\nOPPE Schedule (May 2026 Term):\n" + "\n".join(f"- {l}" for l in lines) + "\n\n"

        return _table_to_markdown(table_html)

    return table_pattern.sub(convert_table, text)


def clean_markdown(raw_text: str) -> str:
    text = expand_table_links(raw_text)  # resolve Table 2/3 anchors before URL stripping
    noise_patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}\s+[AP]M[^\n]*',
        r'Google Docs(?:\s+icon|\s+logo)?\s+Published using Google Docs[^\n]*',
        r'\d+/\d+\s+info\s+icon\s+Published\s+using\s+Google\s+Docs[^\n]*',
        r'Published\s+using\s+Google\s+Docs[^\n]*',
        r'\binfo\s+icon\b[^\n]*',
        r'IITM\s+BS\s+Degree\s+Programme\s*[-–]\s*Student\s+Hand\w*[^\n]*',
        r'Report\s+abuse\s+Learn\s+more[^\n]*',
        r'Updated\s+automatically\s+every\s+\d+\s+minutes[^\n]*',
        r'https://docs\.google\.com/document/[^\s]+\.{3,}[^\n]*',  # truncated self-referential doc URLs
        r'(?<=\n)https://docs\.google\.com/document/\S+(?=\s*\n\s*\d{1,3}/\d{2,3})',  # doc URL followed by page counter
        r'(?<![\d>=])\b\d{1,3}/(?!100\b)\d{2,3}\b(?!\d)\s*(?=\n|$)',
        r'^#{1,3}\s*BS-DS_\s*May\s*2026\s*Grading\s*document\s*\(Student\)\s*$',
        r'^BS-DS_\s*May\s*2026\s*Grading\s*document\s*\(Student\)\s*$',
        r'^Updated\s+automatically\s+every\s+\d+\s+minutes\s*$',
        r'^#\s+BS-DS_[^\n]*Grading\s+document[^\n]*$',
        r'^BS-DS_[^\n]*Grading\s+document[^\n]*$',
        r'^\d{1,3}/\d{2,3}\s*$',
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

    text = html_tables_to_markdown(text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    text = re.sub(r'([a-zA-Z,])\n([a-z])', r'\1 \2', text)
    return text


def load_and_split(md_path: Path) -> list[Document]:
    raw = md_path.read_text(encoding="utf-8")
    cleaned = clean_markdown(raw)

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
        return_each_line=False,
    )
    header_docs = header_splitter.split_text(cleaned)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    final_docs = char_splitter.split_documents(header_docs)

    for doc in final_docs:
        header_parts = [
            doc.metadata.get("h1", ""),
            doc.metadata.get("h2", ""),
            doc.metadata.get("h3", ""),
        ]
        header_path = " > ".join(p for p in header_parts if p)
        if header_path:
            doc.page_content = (
                f"[Course: {header_path}]\n"
                f"{doc.page_content}\n"
                f"[/Course: {header_path}]"
            )

    for doc in final_docs:
        doc.metadata["source"] = md_path.name

    print(f"  {md_path.name}: {len(final_docs)} chunks")
    return final_docs


def build_index():
    print(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    all_docs: list[Document] = []
    for md_path in sorted(DATA_DIR.glob("*.md")):
        print(f"\nProcessing: {md_path.name}")
        all_docs.extend(load_and_split(md_path))

    if not all_docs:
        print("No valid documents found. Exiting.")
        return

    target = "Chroma Cloud" if USE_CHROMA_CLOUD else f"local dir '{DB_DIR}'"
    print(f"\nEmbedding and indexing {len(all_docs)} chunks into {target}...")

    client = get_chroma_client()
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION_NAME,
    )

    print(f"\n✅ Ingestion complete! {vectorstore._collection.count()} chunks stored.")


if __name__ == "__main__":
    build_index()