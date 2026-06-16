import re
from dotenv import load_dotenv
import os
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv()

DB_DIR          = "db"
COLLECTION_NAME = "handbook_docs"
EMBED_MODEL     = "BAAI/bge-small-en-v1.5"
RERANK_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2" # cross-encoder/ms-marco-MiniLM-L-6-v2

HYBRID_TOP_K    = 8   # candidates fetched by each of vector + BM25
FINAL_TOP_N     = 5    # chunks returned after reranking

# Weights for RRF fusion: [vector_weight, bm25_weight]
ENSEMBLE_WEIGHTS = [0.6, 0.4]

CHROMA_API_KEY  = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT   = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
USE_CHROMA_CLOUD = bool(CHROMA_API_KEY)


def get_chroma_client():
    if USE_CHROMA_CLOUD:
        return chromadb.CloudClient(
            api_key=CHROMA_API_KEY,
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
        )
    return chromadb.PersistentClient(path=DB_DIR)


_NOISE_PATTERNS = [
    re.compile(r'\d+/\d+\s+info\s+icon\s+Published\s+using\s+Google\s+Docs[^\n]*', re.I),
    re.compile(r'Published\s+using\s+Google\s+Docs[^\n]*', re.I),
    re.compile(r'\binfo\s+icon\b[^\n]*', re.I),
    re.compile(r'IITM\s+BS\s+Degree\s+Programme\s*[-–]\s*Student\s+Hand\w*[^\n]*', re.I),
    re.compile(r'Report\s+abuse\s+Learn\s+more[^\n]*', re.I),
    re.compile(r'Updated\s+automatically\s+every\s+\d+\s+minutes[^\n]*', re.I),
    re.compile(r'https://docs\.google\.com/\S+'),
    re.compile(r'(?<![\d>=])\b\d{1,3}/(?!100\b)\d{2,3}\b(?!\d)\s*(?=\n|$)'),
    re.compile(r'^#{1,3}\s*BS-DS_\s*May\s*2026\s*Grading\s*document\s*\(Student\)\s*$', re.I | re.M),
]

def _scrub_noise(text: str) -> str:
    """Remove Google Docs boilerplate fragments from a retrieved chunk."""
    for pat in _NOISE_PATTERNS:
        text = pat.sub('', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _is_toc_chunk(text: str) -> bool:
    """Detect Table of Contents or isolated index pages."""
    t = text.lower()
    return ".........." in t or len(re.findall(r"\d+\.\d+\.\d+", t)) > 3


class Retriever:
    def __init__(self):
        backend = "Chroma Cloud" if USE_CHROMA_CLOUD else f"local dir '{DB_DIR}'"
        print(f"Connecting to Chroma backend: {backend}")

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )

        vectorstore = Chroma(
            client=get_chroma_client(),
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": HYBRID_TOP_K},
        )

        print("Building BM25 index from stored documents...")
        all_docs_data = vectorstore.get()    # returns {"ids", "documents", "metadatas"}
        bm25_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(
                all_docs_data["documents"], all_docs_data["metadatas"]
            )
        ]
        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = HYBRID_TOP_K

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=ENSEMBLE_WEIGHTS,
        )

        self.cross_encoder = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)

        print(f"✅ Retriever ready ({len(bm25_docs)} chunks indexed)")

    def retrieve(self, query: str, top_n: int = FINAL_TOP_N) -> list[dict]:
        """
        Run the full hybrid + rerank pipeline.
        Returns a list of dicts with keys: text, source, page, rerank_score.
        """
        # hybrid fetch (RRF-fused vector + BM25)
        candidates: list[Document] = self.ensemble_retriever.invoke(query)

        # cross-encoder rerank — score every (query, chunk) pair
        pairs  = [(query, doc.page_content) for doc in candidates]
        scores = self.cross_encoder.score(pairs)   # returns list[float]

        # apply ToC penalty, build result dicts, sort by score
        results = []
        for doc, score in zip(candidates, scores):
            if _is_toc_chunk(doc.page_content):
                score -= 5.0

            section_parts = [
                doc.metadata.get("h1", ""),
                doc.metadata.get("h2", ""),
                doc.metadata.get("h3", ""),
            ]
            section = " › ".join(p for p in section_parts if p) or "—"

            results.append({
                "text":         _scrub_noise(doc.page_content),
                "source":       doc.metadata.get("source", "unknown"),
                "page":         section,
                "rerank_score": float(score),
            })

        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_n]


if __name__ == "__main__":
    r = Retriever()

    queries = [
        "What is the alumni fees?",
        "What is the criteria to pass DBMS OPPE?",
        "What is the end term exam date?",
    ]

    for q in queries:
        print(f"\n{'-'*60}\nQuery: {q}\n{'-'*60}")
        chunks = r.retrieve(q)
        for i, c in enumerate(chunks, 1):
            preview = c["text"].replace("\n", " ")[:500]
            print(f"[{i}] {c['source']}  score={c['rerank_score']:.3f}")
            print(f"     Section: {c['page']}")
            print(f"     {preview}...\n")