# IITM BS Degree RAG Assistant

A production-style Retrieval-Augmented Generation (RAG) system that answers
student questions about the IITM BS Degree Programme — fees, grading
policies, exam schedules, eligibility rules, and academic structure — strictly
grounded in the official student handbook and grading documents, with
citations back to the source section.

This isn't a "load a PDF, ask a question" demo. It's built around the gap
between a RAG *demo* and a RAG *system*: hybrid retrieval, reranking,
versioned prompts, an automated evaluation harness, and a CI quality gate.

---

## Architecture

```
data/*.md  →  ingest.py  →  ChromaDB (vector store)
                                  │
question  →  retrieve.py  ───────┴─── BM25 (keyword index)
                  │
          EnsembleRetriever (RRF fusion, weights 0.6/0.4)
                  │
          Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
                  │
          top-N chunks (with source + section metadata)
                  │
generate.py  →  prompts.yaml (versioned system/human prompts)
                  │
          Groq LLM (llama-3.3-70b-versatile)
                  │
          Grounded answer + cited sources
```

### Ingestion (`src/ingest.py`)
- Source documents are Google-Docs-exported Markdown (`student-handbook.md`,
  `grading-document.md`)
- Custom cleaning step strips Google Docs export noise (page numbers, "Published
  using Google Docs" footers, info-icon artifacts) and converts embedded HTML
  tables to Markdown pipe tables
- Header-aware splitting (`MarkdownHeaderTextSplitter` on `#`/`##`/`###`) followed
  by `RecursiveCharacterTextSplitter` (2000 char chunks, 200 char overlap)
- Each chunk is wrapped with `[Course: h1 > h2 > h3]` tags so a chunk that lost
  its header during splitting still surfaces in both vector and BM25 search
- Embeddings: `BAAI/bge-small-en-v1.5`, stored in ChromaDB

### Retrieval (`src/retrieve.py`)
- **Hybrid search**: vector similarity + BM25, fused via `EnsembleRetriever`
  (weights `[0.6, 0.4]`), 15 candidates from each
- **Cross-encoder reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` rescores
  every (query, chunk) pair
- **ToC penalty**: chunks that look like a table of contents (dotted leaders,
  repeated `x.y.z` numbering) get a score penalty so they don't crowd out real
  content
- **Noise scrubbing**: a second regex pass removes any residual export
  artifacts from retrieved chunks before they reach the LLM
- Returns the top N chunks with `text`, `source`, `page` (section path), and
  `rerank_score`

### Generation (`src/generate.py`)
- Prompts are **not hardcoded** — they're loaded from `prompts.yaml` at
  startup and printed with their version number
- The system prompt enforces: strict grounding (no hallucination, explicit
  "I don't have that information" fallback), careful table extraction,
  mandatory source citations, defining acronyms/fee terms before using them,
  flagging potentially-overlapping fee figures, and focusing on one course at
  a time
- Streams the answer token-by-token and prints the deduplicated source list
  used for the response

---

## Evaluation (`eval/evaluate.py`)

A 10-question golden set (`eval/eval_prompts.json`), curated for **category
coverage** rather than size — schedules, grading policy, eligibility, fees,
administrative, minors, certification, and academic structure are all
represented. Questions that score poorly are deliberately kept in rather than
dropped, since the eval is meant to surface weak spots, not flatter the score.

**Pipeline per question:**
1. Retrieve top-4 chunks via the full hybrid + rerank pipeline
2. Generate an answer using `llama-3.3-70b-versatile` (the *eval_system*
   prompt + shared `human` template from `prompts.yaml`, capped at 150 tokens)
3. Fast keyword-hit check (no LLM) as a cheap correctness signal
4. Three DeepEval LLM-judge metrics, also via `llama-3.3-70b-versatile`:
   - **Faithfulness** — are the answer's claims supported by the retrieved chunks?
   - **Answer Relevancy** — does the answer address the question asked?
   - **Contextual Precision** — are the most useful chunks ranked highest?

**Reliability features:**
- Automatic retry with backoff on Groq 429s, parsing the suggested wait time
  from the error message
- Throttling (`BETWEEN_CALLS=3s`, `BETWEEN_METRICS=4s`) to stay under rate limits
- Per-question checkpointing (`eval_checkpoint.json`) — a crash mid-run resumes
  from where it left off instead of restarting; the checkpoint is cleared on a
  clean pass
- Every run writes `eval/report.json` with the model versions, prompts
  version, per-metric averages, per-question breakdown, and pass/fail gate

**Latest result** (10 questions, threshold 0.5):

| Metric | Score |
|---|---|
| Keyword Hit Rate | 100% |
| Faithfulness (avg) | 0.933 |
| Answer Relevancy (avg) | 0.883 |
| Contextual Precision (avg) | 0.817 |
| **Gate** (min of keyword & faithfulness) | **0.933 — PASSED** |

A note on judge choice: an earlier run using `llama-3.1-8b-instant` as the
judge produced a gate score of 0.618, with several faithfulness scores of
0.000 on answers that were manually verified as correct. The 8B judge lacks
the reasoning capacity for this domain's compound eligibility/fee questions.
Switching the judge to 70B (keeping generation on 70B as well) raised the gate
to 0.933 — reflecting the RAG system's actual quality rather than judge error.

---

## Prompt versioning (`prompts.yaml`)

Prompts are treated as part of the system's configuration, not inline strings:

- `system` — the full production system prompt used by `generate.py`
- `eval_system` — a leaner version used by `evaluate.py`, sized for the
  150-token eval budget while preserving the core grounding/refusal behavior
  that the faithfulness metric actually scores
- `human` — the shared `{context}` / `{question}` template used by both
  production and eval, so retrieval-context formatting stays identical
- `version` — bumped whenever any prompt changes; recorded in every
  `report.json` so a score can always be traced back to the exact prompt that
  produced it

---

## CI (`.github/workflows/ci.yaml`)

On every push, GitHub Actions installs dependencies, rebuilds the vector index
from `data/*.md`, and runs the evaluation gate. Currently configured for a
fast keyword-only gate (`--no-deepeval`); see "What's next" below for wiring in
the full DeepEval gate.

---

## Project layout

```
RAG/
├── data/                    # source Markdown documents
├── src/
│   ├── ingest.py            # chunk + embed + index
│   ├── retrieve.py          # hybrid search + rerank
│   └── generate.py          # RAG chain + CLI chat
├── eval/
│   ├── eval_prompts.json    # golden question set
│   ├── evaluate.py           # DeepEval harness
│   ├── report.json           # latest run's results
│   └── eval_checkpoint.json  # resume state (cleared on pass)
├── prompts.yaml             # versioned prompts (prod + eval)
├── requirements.txt
└── .github/workflows/ci.yaml
```

## Running it

```bash
pip install -r requirements.txt

# Build the index
python src/ingest.py

# Chat with the assistant
python src/generate.py

# Run the eval suite
python eval/evaluate.py --save-report
```

Requires a `GROQ_API_KEY` in `.env`.

## Tech stack

LangChain (LCEL), ChromaDB, BM25 (`rank-bm25`), `BAAI/bge-small-en-v1.5`
embeddings, `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker, Groq
(`llama-3.3-70b-versatile`), DeepEval, GitHub Actions.

## What's next

- Wire the full DeepEval faithfulness gate into CI (currently keyword-only),
  likely scoped to PRs into `main` given the ~3-5 minute runtime with throttling
- Add an observability/tracing layer (latency breakdown, cost per request,
  citation coverage, failure rate) on top of this pipeline