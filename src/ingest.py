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
    noise_patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}\s+[AP]M[^\n]*',
        r'Google Docs(?:\s+icon|\s+logo)?\s+Published using Google Docs[^\n]*',
        r'\d+/\d+\s+info\s+icon\s+Published\s+using\s+Google\s+Docs[^\n]*',
        r'Published\s+using\s+Google\s+Docs[^\n]*',
        r'\binfo\s+icon\b[^\n]*',
        r'IITM\s+BS\s+Degree\s+Programme\s*[-–]\s*Student\s+Hand\w*[^\n]*',
        r'Report\s+abuse\s+Learn\s+more[^\n]*',
        r'Updated\s+automatically\s+every\s+\d+\s+minutes[^\n]*',
        r'https://docs\.google\.com/\S+',
        r'(?<![\d>=])\b\d{1,3}/(?!100\b)\d{2,3}\b(?!\d)\s*(?=\n|$)',
        r'^#{1,3}\s*BS-DS_\s*May\s*2026\s*Grading\s*document\s*\(Student\)\s*$',
        r'^BS-DS_\s*May\s*2026\s*Grading\s*document\s*\(Student\)\s*$',
        r'^Updated\s+automatically\s+every\s+\d+\s+minutes\s*$',
    ]
    text = raw_text
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

    text = html_tables_to_markdown(text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
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