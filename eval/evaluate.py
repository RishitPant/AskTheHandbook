
import os
import sys
import json
import argparse
import time
import re
from datetime import datetime
from pathlib import Path
import yaml

from dotenv import load_dotenv
from groq import Groq
from groq import RateLimitError
from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from retrieve import Retriever

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    print("ERROR: GROQ_API_KEY not found in environment.")
    sys.exit(1)

EVAL_DATA_PATH    = Path(__file__).parent / "eval_prompts.json"
REPORT_PATH       = Path(__file__).parent / "report.json"
CHECKPOINT_PATH   = Path(__file__).parent / "eval_checkpoint.json"
DEFAULT_THRESHOLD = 0.5
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
GEN_MODEL   = os.getenv("RAG_MODEL",   "llama-3.3-70b-versatile") 

PROMPTS_PATH = ROOT / "prompts.yaml"
if not PROMPTS_PATH.exists():
    print(f"ERROR: prompts.yaml not found at {PROMPTS_PATH}")
    sys.exit(1)

_prompts        = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))
PROMPTS_VERSION = _prompts.get("version", "unknown")
EVAL_SYSTEM     = _prompts["eval_system"]
HUMAN_TEMPLATE  = _prompts["human"]

# Retry / throttle settings
MAX_RETRIES     = 6
BACKOFF_BASE    = 2   # seconds — used only if retry delay isn't parseable
BETWEEN_CALLS   = 3   # polite gap after every successful Groq gen call
BETWEEN_METRICS = 4   # gap between each metric.measure() judge call


# ---- Retry helper ---- 

def _parse_retry_delay(error: RateLimitError) -> float | None:
    msg = str(error)

    # milliseconds: "760ms"
    ms_match = re.search(r'try again in (\d+(?:\.\d+)?)ms', msg, re.I)
    if ms_match:
        return float(ms_match.group(1)) / 1000.0 + 0.5

    # seconds: "1.2s"
    s_match = re.search(r'try again in (\d+(?:\.\d+)?)s', msg, re.I)
    if s_match:
        return float(s_match.group(1)) + 0.5

    return None


def groq_call_with_retry(fn, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = fn(*args, **kwargs)
            time.sleep(BETWEEN_CALLS)
            return result
        except RateLimitError as e:
            if attempt == MAX_RETRIES:
                raise

            suggested = _parse_retry_delay(e)
            wait = suggested if suggested else (BACKOFF_BASE ** attempt)
            print(f"\n    ⏳ 429 rate-limited — waiting {wait:.2f}s (attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)


# ── Groq wrapper for DeepEval 

class GroqJudge(DeepEvalBaseLLM):
    """
    Wraps the Groq SDK so DeepEval can use it as its judge LLM.

    DeepEval calls generate() with either:
      - just a prompt string       → return a plain string
      - a prompt + Pydantic schema → return a parsed schema instance
    """
    def __init__(self, api_key: str, model_name: str = JUDGE_MODEL):
        self.api_key    = api_key
        self.model_name = model_name
        self._client    = Groq(api_key=api_key)

    def load_model(self):
        return self._client

    def generate(self, prompt: str, schema: BaseModel = None):
        client = self.load_model()
        kwargs = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        response = groq_call_with_retry(client.chat.completions.create, **kwargs)
        raw = response.choices[0].message.content.strip()

        if schema is not None:
            for candidate in [raw] + raw.split("```"):
                candidate = candidate.lstrip("json").strip()
                try:
                    return schema(**json.loads(candidate))
                except Exception:
                    continue
            raise ValueError(f"GroqJudge: could not parse schema: {raw[:200]}")

        return raw

    async def a_generate(self, prompt: str, schema: BaseModel = None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"Groq/{self.model_name}"


# ── Answer generator

def generate_answer(question: str, chunks: list[dict], client: Groq) -> str:
    context_parts = [
        f"[{c['source']} — Section: {c['page']}]\n{c['text']}"
        for c in chunks
    ]
    context     = "\n---\n".join(context_parts)
    user_prompt = HUMAN_TEMPLATE.format(context=context, question=question)

    response = groq_call_with_retry(
        client.chat.completions.create,
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": EVAL_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


# ── Keyword hit

def keyword_hit(answer: str, expected_keywords: list[str]) -> bool:
    """Check if any expected keyword appears in the answer (case-insensitive)."""
    a = answer.lower()
    return any(kw.lower() in a for kw in expected_keywords)


# ── Checkpoint helpers

def _load_checkpoint() -> dict:
    """Return previously saved per-question scores, keyed by question id."""
    if CHECKPOINT_PATH.exists():
        try:
            return json.loads(CHECKPOINT_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_checkpoint(data: dict) -> None:
    CHECKPOINT_PATH.write_text(json.dumps(data, indent=2))



def run_evaluation(
    category: str      = None,
    use_deepeval: bool = True,
    threshold: float   = DEFAULT_THRESHOLD,
    save_report: bool  = True,
):
    print("\n" + "=" * 65)
    print("  AskTheHandbook — DEEPEVAL EVALUATION")
    print("=" * 65)

    if not EVAL_DATA_PATH.exists():
        print(f"ERROR: eval_prompts.json not found at {EVAL_DATA_PATH}")
        sys.exit(1)

    with open(EVAL_DATA_PATH) as f:
        eval_data = json.load(f)

    if category:
        eval_data = [q for q in eval_data if q.get("category") == category]
        print(f"  Category filter : '{category}' → {len(eval_data)} questions")
    else:
        print(f"  Total questions : {len(eval_data)}")

    print(f"  Judge model     : {JUDGE_MODEL}")
    print(f"  Gen model       : {GEN_MODEL}")
    print(f"  Prompts version : {PROMPTS_VERSION}")
    print(f"  DeepEval        : {'enabled' if use_deepeval else 'disabled (keyword-only)'}")
    print(f"  Threshold       : {threshold}\n")

    print("Initializing retriever...")
    retriever   = Retriever()
    groq_client = Groq(api_key=API_KEY)

    print("\nPhase 1 — Retrieve & Generate\n" + "-" * 40)

    test_cases  : list[LLMTestCase] = []
    kw_hits     : list[bool]        = []
    item_map    : list[dict]        = []
    chunk_scores: list[list[float]] = []

    for i, item in enumerate(eval_data, 1):
        question = item["question"]
        print(f"  [{i:02d}/{len(eval_data)}] {question[:70]}")

        chunks   = retriever.retrieve(question, top_n=4)
        answer   = generate_answer(question, chunks, groq_client)
        contexts = [c["text"][:1000] for c in chunks]
        scores   = [round(c["rerank_score"], 3) for c in chunks]

        kw = keyword_hit(answer, item["expected_keywords"])
        kw_hits.append(kw)
        chunk_scores.append(scores)

        print(f"         rerank scores : {scores}")
        print(f"         keyword       : {'✅' if kw else '❌'}  {answer[:80]}{'…' if len(answer) > 80 else ''}\n")

        test_cases.append(LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=contexts,
            expected_output=" | ".join(item["expected_keywords"]),
        ))
        item_map.append(item)

    kw_rate = sum(kw_hits) / len(kw_hits)

    # DeepEval scoring 
    results_by_metric   : dict[str, list[float]] = {}
    per_question_scores : list[dict]             = []

    if use_deepeval:
        print("\nPhase 2 — DeepEval Metrics\n" + "-" * 40)
        print(f"  Judge model       : {JUDGE_MODEL}")
        print(f"  Gap between calls : {BETWEEN_CALLS}s  |  Max retries on 429 : {MAX_RETRIES}\n")

        judge = GroqJudge(api_key=API_KEY)

        metrics = [
            FaithfulnessMetric(
                threshold=threshold, model=judge,
                include_reason=True, async_mode=False,
            ),
            AnswerRelevancyMetric(
                threshold=threshold, model=judge,
                include_reason=True, async_mode=False,
            ),
            ContextualPrecisionMetric(
                threshold=threshold, model=judge,
                include_reason=True, async_mode=False,
            ),
        ]

        checkpoint = _load_checkpoint()
        if checkpoint:
            print(f"  📂 Resuming from checkpoint — {len(checkpoint)} question(s) already done\n")

        for i, (tc, item) in enumerate(zip(test_cases, item_map), 1):
            qid = item["id"]
            print(f"  [{i:02d}/{len(test_cases)}] {tc.input[:65]}")

            if qid in checkpoint:
                q_scores = checkpoint[qid]
                print(f"    ↩️  skipped (checkpoint)\n")
                for mname, score in q_scores.items():
                    if mname in ("question", "keyword_hit"):
                        continue
                    results_by_metric.setdefault(mname, []).append(score)
                per_question_scores.append(q_scores)
                continue

            q_scores = {"question": tc.input, "keyword_hit": kw_hits[i - 1]}

            for m in metrics:
                mname = type(m).__name__
                try:
                    m.measure(tc)
                    score  = m.score if m.score is not None else 0.0
                    reason = (m.reason or "—")[:110]
                    icon   = "✅" if score >= threshold else "❌"
                    print(f"    {mname:<32} {icon} {score:.3f}  {reason}")
                except RateLimitError as e:
                    score = 0.0
                    print(f"    {mname:<32} ⚠️  rate limit exhausted after {MAX_RETRIES} retries: {e}")
                except Exception as e:
                    score = 0.0
                    print(f"    {mname:<32} ⚠️  error: {e}")

                results_by_metric.setdefault(mname, []).append(score)
                q_scores[mname] = round(score, 4)

                # Polite gap between judge calls to avoid 429s
                time.sleep(BETWEEN_METRICS)

            per_question_scores.append(q_scores)
            checkpoint[qid] = q_scores
            _save_checkpoint(checkpoint)   # flush after every question
            print()

    # Aggregate summary
    print("=" * 65)
    print("  AGGREGATE RESULTS")
    print("=" * 65)
    print(f"  Questions evaluated        : {len(eval_data)}")
    print(f"  Keyword Hit Rate           : {kw_rate:.1%}  {'✅' if kw_rate >= threshold else '❌'}")

    def _avg(lst: list[float]) -> float:
        valid = [s for s in lst if s is not None]
        return sum(valid) / len(valid) if valid else 0.0

    avg_faith = avg_rel = avg_prec = None

    if use_deepeval and results_by_metric:
        avg_faith = _avg(results_by_metric.get("FaithfulnessMetric",       []))
        avg_rel   = _avg(results_by_metric.get("AnswerRelevancyMetric",    []))
        avg_prec  = _avg(results_by_metric.get("ContextualPrecisionMetric",[]))

        print(f"  Faithfulness (avg)         : {avg_faith:.3f}  {'✅' if avg_faith >= threshold else '❌'}")
        print(f"  Answer Relevancy (avg)     : {avg_rel:.3f}  {'✅' if avg_rel   >= threshold else '❌'}")
        print(f"  Contextual Precision (avg) : {avg_prec:.3f}  {'✅' if avg_prec  >= threshold else '❌'}")

        print("\n  Per-question breakdown:")
        header = f"  {'ID':<28} {'kw':>3}  {'Faith':>6}  {'Rel':>6}  {'Prec':>6}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for item, kw, pq in zip(item_map, kw_hits, per_question_scores):
            f = pq.get("FaithfulnessMetric",       0)
            r = pq.get("AnswerRelevancyMetric",     0)
            p = pq.get("ContextualPrecisionMetric", 0)
            print(
                f"  {item['id']:<28} {'✅' if kw else '❌':>3} "
                f" {f:>6.3f}  {r:>6.3f}  {p:>6.3f}"
            )

    # ── CI gate 
    if avg_faith is not None:
        gate_metric = min(kw_rate, avg_faith)
        gate_label  = f"min(keyword={kw_rate:.1%}, faithfulness={avg_faith:.3f})"
    else:
        gate_metric = kw_rate
        gate_label  = f"keyword hit rate = {kw_rate:.1%}"

    print(f"\n  Gate  : {gate_label}")
    print(f"  Score : {gate_metric:.3f}  (threshold: {threshold:.2f})")

    # ── Optional JSON report 
    if save_report:
        report = {
            "timestamp":        datetime.now().isoformat(),
            "judge_model":      JUDGE_MODEL,
            "gen_model":        GEN_MODEL,
            "prompts_version":  PROMPTS_VERSION,
            "threshold":        threshold,
            "category":         category,
            "num_questions":    len(eval_data),
            "keyword_hit_rate": round(kw_rate, 4),
            "averages": {
                "faithfulness":         round(avg_faith, 4) if avg_faith is not None else None,
                "answer_relevancy":     round(avg_rel,   4) if avg_rel   is not None else None,
                "contextual_precision": round(avg_prec,  4) if avg_prec  is not None else None,
            },
            "gate_score":  round(gate_metric, 4),
            "passed":      gate_metric >= threshold,
            "per_question": per_question_scores,
        }
        REPORT_PATH.write_text(json.dumps(report, indent=2))
        print(f"\n  📄 Report saved → {REPORT_PATH}")

    # Exit with CI-friendly code
    if gate_metric >= threshold:
        print(f"\n  ✅ PASSED — RAG quality is above threshold ({threshold:.0%})\n")
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
            print("  🗑️  Checkpoint cleared.\n")
        sys.exit(0)
    else:
        print(f"\n  ❌ FAILED — Quality dropped below threshold ({threshold:.0%})")
        print("     Check ❌ rows above. Re-run ingest.py if documents changed.\n")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AskTheHandbook with DeepEval")
    parser.add_argument("--category",    type=str,   default=None,
                        help="Filter eval_prompts.json by category field")
    parser.add_argument("--threshold",   type=float, default=DEFAULT_THRESHOLD,
                        help="Pass/fail threshold for all metrics (default: 0.5)")
    parser.add_argument("--no-deepeval", action="store_true",
                        help="Skip DeepEval metrics; run keyword check only")
    parser.add_argument("--save-report", action="store_true",
                        help="Write results to eval/report.json")
    args = parser.parse_args()

    run_evaluation(
        category=args.category,
        use_deepeval=not args.no_deepeval,
        threshold=args.threshold,
        save_report=True,
    )