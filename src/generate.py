import os
import sys
from dotenv import load_dotenv
from pathlib import Path
import yaml
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from retrieve import Retriever

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
HUMAN_PROMPT    = _prompts["H"]

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
        llm = ChatGroq(
            model=MODEL, #llama-3.3-70b-versatile
            temperature=0.1,
            streaming=True,
        )

        # LCEL chain:
        # 1. Retrieve chunks for the question
        # 2. Format them into a context string
        # 3. Pass context + question into the prompt
        # 4. Send to LLM and parse output
        self.chain = (
            {
                "context":  RunnableLambda(lambda q: format_context(self.retriever.retrieve(q))),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # Keep a reference so answer() can show sources
        self._last_chunks: list[dict] = []

    def answer(self, query: str, top_n: int = 4) -> str:
        """Run the full RAG chain and stream the answer to stdout."""
        print(f"\n🔍 Retrieving context for: '{query}'...")
        self._last_chunks = self.retriever.retrieve(query, top_n=top_n)

        if not self._last_chunks:
            msg = "I couldn't find any official documentation related to that question."
            print(msg)
            return msg

        print("🧠 Generating answer...\n")

        full_response = ""
        for token in self.chain.stream(query):
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

        return full_response

if __name__ == "__main__":
    agent = Generator()

    print("\n🎓 IITM BS Degree RAG Assistant Online!")
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
