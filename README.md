---
title: AskTheHandbook
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# AskTheHandbook

A production-style RAG system that answers student questions about the IITM BS Degree Programme (fees, grading, exams, eligibility, academic structure) strictly grounded in the official handbook and grading documents, with source citations.

🔗 **Live demo:** [huggingface.co/spaces/rishitpant/AskTheHandbook](https://huggingface.co/spaces/rishitpant/AskTheHandbook)

Features:
- Hybrid retrieval + cross-encoder reranking
- Versioned prompts
- Automated evaluation harness with checkpointing
- Langfuse observability
- Rate limiting via `slowapi` (5 requests/minute per IP, proxy-aware)
- CI quality gate with auto-deploy to Hugging Face Spaces
- FastAPI backend + web chat UI, Dockerized for production

---

## Architecture

```
source PDFs  →  LlamaParse  →  data/*.md  →  ingest.py  →  Chroma Cloud (or local ChromaDB)
                                                                  │
question  →  retrieve.py  ────────────────────────────────────────┴─── BM25 (keyword index)
                  │
          EnsembleRetriever (RRF fusion, weights 0.6/0.4)
                  │
          Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
                  │
          top-N chunks (with source + section metadata)
                  │
generate.py  →  prompts.yaml (versioned prompts)
                  │
          Groq LLM (llama-3.3-70b-versatile), streamed
                  │
          Grounded answer + cited sources
                  │
   app.py (FastAPI) → static/index.html (chat UI) / CLI
                  │
          Langfuse trace (latency, chunk metadata, rerank scores)
```


---

## Project layout

```
RAG/
├── app.py                         # FastAPI app (chat/stream + static UI + rate limiting)
├── static/
│   └── index.html                 # chat frontend
├── data/
│   ├── student-handbook.md        # source document 1
│   └── grading-document.md        # source document 2
├── src/
│   ├── ingest.py                  # chunk + embed + index (local or Chroma Cloud)
│   ├── retrieve.py                # hybrid search + rerank
│   ├── generate.py                # Generator class, RAG chain + CLI chat
│   └── tracing.py                 # Langfuse observability (opt-in)
├── eval/
│   ├── eval_prompts.json          # 10-question set
│   ├── evaluate.py                # DeepEval harness
│   └── report.json                # latest run's results
├── db/                            # local ChromaDB vector store (generated)
├── prompts.yaml                   # versioned prompts (prod + eval)
├── Dockerfile
├── requirements.txt               # full dev environment
├── requirements-prod.txt          # slim deps for Docker/CI
└── .github/workflows/ci.yaml      # eval gate + HF Spaces deploy
```
---

### Ingestion (`src/ingest.py`)
- Source docs: PDFs (exported from Google Docs) converted to Markdown via LlamaParse (`student-handbook.md`, `grading-document.md`)
- Cleans residual Google Docs export noise (page numbers, "Published using Google Docs" footers, info-icon artifacts) carried over from the source PDFs; converts HTML tables to Markdown tables
- Header-aware splitting (`MarkdownHeaderTextSplitter`) + `RecursiveCharacterTextSplitter` (2000 char chunks, 200 overlap)
- Wraps each chunk with `[Course: h1 > h2 > h3]` tags for both vector and BM25 search
- Embeddings: `BAAI/bge-small-en-v1.5`
- Writes to **Chroma Cloud** if `CHROMA_API_KEY` is set, else to local ChromaDB

### Retrieval (`src/retrieve.py`)
- Hybrid search: vector + BM25, fused via `EnsembleRetriever` (weights `[0.6, 0.4]`), 8 candidates each
- Cross-encoder reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2` rescores every (query, chunk) pair
- ToC penalty: deprioritizes table-of-contents-like chunks
- Scrubs residual export noise before chunks reach the LLM
- Returns top N chunks (default 5) with `text`, `source`, `page` (section path), `rerank_score`
- Same Chroma Cloud / local switch as ingestion, via `USE_CHROMA_CLOUD`

### Generation (`src/generate.py`)
- `Generator` class wraps the retriever + Groq LLM client, used by both the CLI and the API
- Prompts loaded from `prompts.yaml` (not hardcoded), version printed at startup
- System prompt enforces: strict grounding, no hallucination, table extraction, mandatory citations, defining acronyms, flagging overlapping fee figures, one course at a time
- Streams tokens via Groq's OpenAI-compatible endpoint
- Model overridable via `RAG_MODEL` env var (default: `llama-3.3-70b-versatile`)
- `get_sources()` returns the deduplicated source list for the last answer
- Each turn wrapped in an `@observe` span for tracing

### Web app (`app.py`, `static/index.html`)
- FastAPI app exposing:
  - `GET /api/health` — pipeline/backend status
  - `POST /api/chat` — non-streaming chat
  - `POST /api/chat/stream` — Server-Sent Events streaming chat
- RAG pipeline loaded once at startup (FastAPI lifespan), not per-request
- Static chat UI served from `static/` at `/`
- Rate limiting: 5 requests/minute per IP via `slowapi`; proxy-aware (reads `X-Forwarded-For`)
- CORS open by default (`allow_origins=["*"]`)

### Observability (`src/tracing.py`)
- Langfuse tracing is opt-in via `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
- When enabled: each turn produces a trace with a `hybrid_retrieval` span (chunks, sections, rerank scores, model costs, latency)
- When disabled: `observe` becomes a no-op — no tracing dependency required

---

## Evaluation (`eval/evaluate.py`)

- 20-question set (`eval/eval_prompts.json`), curated for category coverage, not size due to api rate limits
- Weak-scoring questions are kept deliberately to surface gaps, not flatter the score

**Pipeline per question:**
- Retrieve top-4 chunks via the full hybrid + rerank pipeline
- Generate an answer with `llama-3.3-70b-versatile` (`eval_system` + shared `human` prompt, 150-token cap)
- Fast keyword-hit check (no LLM) due to api rate limits
- Three DeepEval LLM-judge metrics (same model as judge): Faithfulness, Answer Relevancy, Contextual Precision.

**Reliability features:**
- Auto-retry with backoff on Groq 429s
- Throttling between calls/metrics to stay under rate limits
- Per-question checkpointing (`eval_checkpoint.json`) — resumes after a crash, clears on a clean pass
- Every run writes `eval/report.json`: model/prompt versions, per-metric averages, per-question breakdown, pass/fail gate
- Judge model overridable via `JUDGE_MODEL` env var (default: `llama-3.3-70b-versatile`); CI uses `llama-3.1-8b-instant` for speed

**Latest result** (20 questions, judge = `llama-3.3-70b-versatile`, threshold 0.8):

| Metric | Score |
|---|---|
| Keyword Hit Rate | 100% |
| Faithfulness (avg) | 91% |
| Answer Relevancy (avg) | 91% |
| Contextual Precision (avg) | 96% |
| **Gate** (min of keyword & faithfulness) | **0.908 — PASSED** |

- **Judge choice matters:** an 8B judge produced a gate score of 0.618 with false-zero faithfulness scores on verified-correct answers — it lacks reasoning capacity for compound eligibility/fee questions. The 70B judge raised the gate to 0.908, reflecting actual system quality. CI keeps the 8B model for speed; 70B is used for local runs.

---

## Prompt versioning (`prompts.yaml`)

- `system` — full production system prompt (`generate.py`)
- `eval_system` — leaner version for the 150-token eval budget, same core grounding/refusal behavior
- `human` — shared `{context}` / `{question}` template for both prod and eval
- `version` — bumped on any prompt change, recorded in every `report.json`

Current version: **1.4.0**

---

## CI/CD (`.github/workflows/ci.yaml`)

On push to `main`:
1. **Quality gate** — installs `requirements-prod.txt`, runs the keyword-only eval against Chroma Cloud using `llama-3.1-8b-instant` (`python eval/evaluate.py --no-deepeval --threshold 0.8`)
2. **Deploy** — if the gate passes, force-pushes the repo to the linked Hugging Face Space, which rebuilds the Docker image and redeploys

- DeepEval (faithfulness/relevancy/precision) gate is **not yet wired into CI** due to api rate limits — roadmap item, planned for PRs into `main` only (~5–10 min throttled runtime)

---

## Deployment

- **Docker**: `Dockerfile` builds a slim image (`requirements-prod.txt`), runs `uvicorn app:app` on port `7860`
- **Hugging Face Spaces**: repo includes HF Spaces metadata (top of this file); CI auto-deploys on every successful `main` push
- **Vector store in prod**: the image does **not** ship `data/` or a local `db/` — it expects `CHROMA_API_KEY`/`CHROMA_TENANT`/`CHROMA_DATABASE` pointing at a pre-ingested Chroma Cloud collection

---

## How to run

```bash
pip install -r requirements.txt

# Build the vector index (required before first run)
python src/ingest.py

# Chat with the assistant (CLI)
python src/generate.py

# Run the web app locally
uvicorn app:app --reload
# → open http://localhost:8000

# Run the full eval suite (DeepEval + keyword gate)
python eval/evaluate.py --save-report

# Run keyword-only gate (fast, no LLM judge)
python eval/evaluate.py --no-deepeval --threshold 0.5
```

```bash
# Or run via Docker
docker build -t askthehandbook .
docker run -p 7860:7860 --env-file .env askthehandbook
```

**Environment variables** (create a `.env` file):

```
GROQ_API_KEY=your_key_here

# Optional — use Chroma Cloud instead of local persistent ChromaDB
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=...

# Optional — enables Langfuse tracing
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## Tech stack

LangChain (LCEL) · ChromaDB / Chroma Cloud · BM25 (`rank-bm25`) · `BAAI/bge-small-en-v1.5` embeddings · `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker · Groq (`llama-3.3-70b-versatile`) · FastAPI · slowapi · DeepEval · Langfuse · GitHub Actions · Docker · Hugging Face Spaces

---

## What's next

- Wire the DeepEval gate into CI (currently keyword-only; scope to PRs into `main`)
- Improve Contextual Precision on scheduling/fee-overlap questions — tune BM25/vector weights, increase `HYBRID_TOP_K`, or add query expansion
- Add auth to the public API endpoints before wider sharing