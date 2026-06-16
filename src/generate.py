import os
import sys
from dotenv import load_dotenv
from pathlib import Path
import yaml
from langchain_core.prompts import ChatPromptTemplate
from retrieve import Retriever
from tracing import traced_retrieve, langfuse, LANGFUSE_ENABLED, observe
from langfuse.openai import OpenAI

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY environment variable not found.")
    sys.exit(1)

PROMPTS_PATH = Path(__file__).parent.parent / "prompts.yaml"
if not PROMPTS_PATH.exists():
    print(f"ERROR: prompts.yaml not found at {PROMPTS_PATH}")
    sys.exit(1)

_prompts = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))
PROMPTS_VERSION = _prompts.get("version", "unknown")
SYSTEM_PROMPT   = _prompts["system"]
HUMAN_PROMPT    = _prompts["human"]

print(f"Loaded prompts version: {PROMPTS_VERSION}")

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  HUMAN_PROMPT),
])


def format_context(chunks: list[dict]) -> str:
    """Convert retrieved chunk dicts into a formatted context string."""
    parts = []
    for i, c in enumerate(chunks, 1):
        tag = f"[{c['source']} — {c['page']}]"
        parts.append(f"--- Chunk {i} {tag} ---\n{c['text']}")
    return "\n\n".join(parts)


class Generator:
    def __init__(self):
        print("Initializing retrieval system...")
        self.retriever = Retriever()

        MODEL = os.getenv("RAG_MODEL", "llama-3.3-70b-versatile")
        self.model_name = MODEL
        self.llm_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY")
        )

        self._last_chunks: list[dict] = []

    def get_sources(self) -> list[dict]:
        seen = set()
        sources = []
        for c in self._last_chunks:
            key = (c['source'], c['page'])
            if key not in seen:
                seen.add(key)
                sources.append({"source": c['source'], "section": c['page']})
        return sources

    @observe(name="rag_chat_turn")
    def answer(self, query: str, top_n: int = 4):
        """Run the full RAG chain and stream the answer to stdout."""
        print(f"\n🔍 Retrieving context for: '{query}'...")
        self._last_chunks = traced_retrieve(self.retriever, query, top_n=top_n)

        if not self._last_chunks:
            yield "I couldn't find any official documentation related to that question."
            return  

        print("🧠 Generating answer...\n")

        context = format_context(self._last_chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": HUMAN_PROMPT.format(context=context, question=query)}
        ]

        full_response = ""
        stream = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.1,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            print(token, end="", flush=True)
            full_response += token

        print("\n\n" + "-" * 60)
        print("SOURCES USED:")
        seen = set()
        for c in self._last_chunks:
            label = f"- {c['source']}  (Section: {c['page'][:60]})"
            if label not in seen:
                print(label)
                seen.add(label)
        langfuse.flush()
        return full_response

if __name__ == "__main__":
    agent = Generator()

    print("\n AskTheHandbook")
    print("Type 'exit' or 'quit' to close.\n")

    while True:
        try:
            user_input = input("\nStudent Question: ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("Shutting down...")
                break
            if not user_input:
                continue
            agent.answer(user_input)
        except KeyboardInterrupt:
            print("\nShutting down...")
            break