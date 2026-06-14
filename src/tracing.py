import os
from dotenv import load_dotenv

load_dotenv()

_REQUIRED_VARS = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
_missing = [v for v in _REQUIRED_VARS if not os.getenv(v)]
LANGFUSE_ENABLED = not _missing

langfuse = None
langfuse_handler = None
observe = None

if LANGFUSE_ENABLED:
    from langfuse import get_client, observe as _observe
    from langfuse.langchain import CallbackHandler

    langfuse = get_client()
    langfuse_handler = CallbackHandler()
    observe = _observe
    print(f"📡 Langfuse tracing enabled → {os.getenv('LANGFUSE_HOST')}")
else:
    # No-op decorator so traced_retrieve still works without Langfuse
    def observe(*args, **kwargs):
        def decorator(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return decorator

    print(f"⚠️  Langfuse tracing disabled — missing env vars: {', '.join(_missing)}")


@observe(name="hybrid_retrieval")
def traced_retrieve(retriever, query: str, top_n: int = 4) -> list[dict]:
    """
    Run the hybrid + rerank retrieval pipeline and, if Langfuse is enabled,
    attach per-chunk metadata (source, section, rerank score) to the span
    so a trace shows exactly what was retrieved and how it was ranked.
    """
    chunks = retriever.retrieve(query, top_n=top_n)

    if LANGFUSE_ENABLED:
        langfuse.update_current_span(
            input={"query": query, "top_n": top_n},
            output={
                "num_chunks": len(chunks),
                "chunks": [
                    {
                        "source":       c["source"],
                        "section":      c["page"],
                        "rerank_score": c["rerank_score"],
                    }
                    for c in chunks
                ],
            },
        )
        langfuse.flush()

    return chunks


def get_callbacks() -> list:
    """Callback list to pass into chain.invoke()/stream() configs."""
    if not LANGFUSE_ENABLED:
        return []
    return [CallbackHandler()]