"""
PHASE 2a: Draft answerable evaluation questions from your indexed chunks.

What it does:
  1. Reads every chunk (text + metadata) back out of the Chroma store.
  2. Samples N_PER_TOPIC substantive chunks per topic (seeded, reproducible).
  3. Asks the LLM to write ONE question + short reference answer per chunk.
  4. Saves eval/questions_draft.json for YOU to review by hand.

These are DRAFTS. Phase 2b is reviewing them: fix, rewrite, or drop each one.

Run from the project root:  python eval/draft_questions.py
"""

import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PERSIST_DIR = str(ROOT / "vectorstore")
COLLECTION_NAME = "medical_guidelines"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT_PATH = ROOT / "eval" / "questions_draft.json"

GROQ_MODEL = "openai/gpt-oss-20b"
N_PER_TOPIC = 10        # ~15 topics -> ~150 drafts; you will keep ~120
MIN_CHUNK_CHARS = 600   # skip tiny chunks
SEED = 42

SYSTEM_PROMPT = """You write evaluation questions for a medical-guideline question-answering system.
Given ONE passage, write ONE question that the passage answers clearly, plus a short reference answer that uses only the passage.

Rules:
- The question must be self-contained. Never say "this passage", "the text", or "the document".
- Do not copy distinctive phrases from the passage. Paraphrase and use everyday wording, the way a patient, student or clinician would ask.
- Ask about a concrete fact, recommendation, threshold, definition or procedure.
- Do not mention the source or title.
- If the passage has no substantive, answerable content (references, boilerplate, fragments), return {"skip": true}.
Return ONLY JSON: {"question": "...", "reference_answer": "..."}"""


def looks_substantive(text: str) -> bool:
    """Cheap filter that drops reference lists and link-heavy chunks."""
    low = text.lower()
    return (
        len(text) >= MIN_CHUNK_CHARS
        and low.count("http") < 3
        and low.count("doi") < 2
    )


def load_chunks() -> list[dict]:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vs = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    data = vs.get(include=["documents", "metadatas"])
    chunks = []
    for text, meta in zip(data["documents"], data["metadatas"]):
        chunks.append({"text": text, **meta})
    return chunks


def ask_llm(client: Groq, topic: str, passage: str) -> dict | None:
    user_msg = f"Topic: {topic}\n\nPassage:\n{passage}"
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.4,
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            out = json.loads(raw)
            if out.get("skip") or "question" not in out:
                return None
            return out
        except Exception as e:  # JSON errors, rate limits, etc.
            print(f"  retry ({attempt + 1}): {e}")
            time.sleep(2)
    return None


def main() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY not set. Check your .env file.")

    random.seed(SEED)
    chunks = [c for c in load_chunks() if looks_substantive(c["text"])]
    if not chunks:
        raise SystemExit("No usable chunks. Did you build the vector store?")

    by_topic = defaultdict(list)
    for c in chunks:
        by_topic[c.get("topic", "unknown")].append(c)

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    drafts = []
    qnum = 1

    for topic in sorted(by_topic):
        pool = by_topic[topic]
        sample = random.sample(pool, min(N_PER_TOPIC, len(pool)))
        print(f"{topic}: drafting from {len(sample)} chunks")
        for c in sample:
            result = ask_llm(client, topic, c["text"])
            time.sleep(0.5)  # be gentle with the free tier
            if result is None:
                continue
            drafts.append(
                {
                    "qid": f"q{qnum:03d}",
                    "type": "answerable",
                    "status": "draft",        # change to "ok" or "drop" when reviewing
                    "rewritten": False,       # set true if you rewrote the question yourself
                    "question": result["question"],
                    "reference_answer": result["reference_answer"],
                    "topic": topic,
                    "source": c.get("source"),
                    "gold_doc_id": c.get("doc_id"),
                    "gold_chunk_id": c.get("chunk_id"),
                    "gold_title": c.get("title"),
                    "gold_passage": c["text"],
                }
            )
            qnum += 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(drafts, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(drafts)} draft questions to {OUT_PATH}")


if __name__ == "__main__":
    main()