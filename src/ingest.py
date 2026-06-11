import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DATA_DIR = Path("data")
DB_DIR = "db"
COLLECTION_NAME = "handbook_docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

HEADERS_TO_SPLIT = [
    ("#",   "h1"),
    ("##",  "h2"),
    ("###", "h3"),
]


def html_tables_to_markdown(text: str) -> str:
    """Convert HTML <table> blocks to Markdown pipe tables."""
    def convert_table(match):
        table_html = match.group(0)
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)
        if not rows:
            return table_html

        md_rows = []
        for row in rows:
            cells = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', row, re.DOTALL | re.IGNORECASE)
            cleaned = [re.sub(r'<[^>]+>', '', cell).strip() for cell in cells]
            if any(cleaned):
                md_rows.append('| ' + ' | '.join(cleaned) + ' |')

        if not md_rows:
            return table_html

        separator = '| ' + ' | '.join(['---'] * len(md_rows[0].split('|')[1:-1])) + ' |'
        md_rows.insert(1, separator)
        return '\n' + '\n'.join(md_rows) + '\n'

    return re.sub(r'<table[^>]*>.*?</table>', convert_table, text,
                  flags=re.DOTALL | re.IGNORECASE)


def clean_markdown(raw_text: str) -> str:
    """Strip Google Docs scrape noise and normalize whitespace."""
    noise_patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}\s+[AP]M[^\n]*',
        r'Google Docs(?: icon| logo)? Published using Google Docs[^\n]*',
        r'Report abuse\s+Learn more[^\n]*',
        r'Updated automatically every \d+ minutes[^\n]*',
        r'https://docs\.google\.com/\S+',
    ]
    text = raw_text
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    text = html_tables_to_markdown(text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def load_and_split(md_path: Path) -> list[Document]:
    raw = md_path.read_text(encoding="utf-8")
    cleaned = clean_markdown(raw)

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
        return_each_line=False
    )
    header_docs = header_splitter.split_text(cleaned)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    final_docs = char_splitter.split_documents(header_docs)
    
    for doc in final_docs:
        doc.metadata["source"] = md_path.name
    
    print(f"{md_path.name}: {len(final_docs)} chunks")
    return final_docs


def build_index():
    print(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    all_docs: list[Document] = []
    for md_path in sorted(DATA_DIR.glob("*md")):
        print(f"\nProcessing: {md_path.name}")
        all_docs.extend(load_and_split(md_path))

    if not all_docs:
        print("No valid documents found. Exiting.")
        return
    
    print(f"\nEmbedding and indexing {len(all_docs)} chunks into ChromaDB...")

    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME
    )

    print(f"\n✅ Ingestion complete! {vectorstore._collection.count()} chunks stored.")


if __name__ == "__main__":
    build_index()