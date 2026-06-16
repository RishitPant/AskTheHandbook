"""
FastAPI entrypoint for the IITM BS Degree RAG Assistant.

Endpoints:
  GET  /api/health        — liveness check, reports Chroma backend in use
  POST /api/chat          — single JSON response {answer, sources}
  POST /api/chat/stream   — Server-Sent Events stream of answer tokens,
                             followed by a final `sources` event

The Retriever (embeddings + vectorstore + cross-encoder) is loaded once at
startup via FastAPI's lifespan, not per-request — model loads are the
expensive part and must not happen on every call.
"""
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent / "src"))

from generate import Generator  # noqa: E402
from retrieve import USE_CHROMA_CLOUD  # noqa: E402

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading RAG pipeline (embeddings, vectorstore, reranker, LLM client)...")
    _state["generator"] = Generator()
    print("✅ RAG pipeline ready.")
    yield
    _state.clear()


app = FastAPI(title="IITM BS Degree RAG Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_generator() -> Generator:
    gen = _state.get("generator")
    if gen is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is still starting up.")
    return gen


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_n: int = Field(4, ge=1, le=10)


class Source(BaseModel):
    source: str
    section: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "chroma_backend": "cloud" if USE_CHROMA_CLOUD else "local",
        "pipeline_ready": "generator" in _state,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    gen = get_generator()
    tokens = list(gen.answer(req.question, top_n=req.top_n))
    return ChatResponse(answer="".join(tokens), sources=gen.get_sources())


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    gen = get_generator()

    def event_stream():
        for token in gen.answer(req.question, top_n=req.top_n):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': gen.get_sources()})}\n\n"  

    return StreamingResponse(
    event_stream(),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    },
)



# Serve the static chat UI (index.html + assets) at the root path.
# Mounted last so /api/* routes above take precedence.
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
