import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

sys.path.insert(0, str(Path(__file__).parent / "src"))

from generate import Generator
from retrieve import USE_CHROMA_CLOUD

_state: dict = {}
limiter = Limiter(key_func=get_remote_address, default_limits=["5/minute"])


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # the header can be a comma-separated chain (client, proxy1, proxy2, ...);
        # the first entry is the original client
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


limiter = Limiter(
    key_func=get_client_ip,
    default_limits=["5/minute"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading RAG pipeline (embeddings, vectorstore, reranker, LLM client)...")
    _state["generator"] = Generator()
    print("✅ RAG pipeline ready.")
    yield
    _state.clear()


app = FastAPI(title="AskTheHandbook", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
@limiter.limit("5/minute")
def chat(req: ChatRequest, request: Request):
    gen = get_generator()
    tokens = list(gen.answer(req.question, top_n=req.top_n))
    return ChatResponse(answer="".join(tokens), sources=gen.get_sources())


@app.post("/api/chat/stream")
@limiter.limit("5/minute")
def chat_stream(req: ChatRequest, request: Request):
    gen = get_generator()

    def event_stream():
        count = 0
        for token in gen.answer(req.question, top_n=req.top_n):
            count += 1
            yield f"data: {json.dumps({'token': token})}\n\n"
        print(f"[STREAM] loop finished, total tokens yielded: {count}", flush=True)
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


static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
