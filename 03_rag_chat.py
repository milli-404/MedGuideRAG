"""
STEP 3: Retrieval-Augmented Generation -- ask questions over the medical
guidelines and get answers grounded in the retrieved text.

What this script does, every time you ask a question:
  1. Embeds your question with the SAME embedding model used in step 2.
  2. Searches Chroma for the top-k most similar chunks (vector search).
  3. Stuffs those chunks into a prompt template as "context".
  4. Sends that prompt to an LLM through a cloud API (this is the
     "cloud/API integration" part of the project).
  5. Prints the answer plus the sources it was grounded in.

Why RAG instead of just asking the LLM directly:
  - The LLM's own training data may be outdated or may not include these
    specific guidelines.
  - By feeding it the actual retrieved text, the answer is grounded in a
    real, citable source instead of the model's memorized (and possibly
    wrong) knowledge -- this hugely reduces hallucination for a domain
    like medicine.

LLM provider: this script uses Groq's API by default because it has a
free tier and serves fast, open-source models (Llama 3.1). Swap
LLM_PROVIDER to "openai" if you'd rather use GPT models -- the retrieval
code doesn't change at all, only the generation call does. That
swappability is the whole point of separating retrieval from generation.
"""

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()  # reads GROQ_API_KEY / OPENAI_API_KEY from a local .env file

PERSIST_DIR = "vectorstore"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4  # how many chunks to retrieve per question

LLM_PROVIDER = "groq"  # "groq" or "openai"
GROQ_MODEL = "openai/gpt-oss-20b"  # Groq deprecated llama-3.1-8b-instant on the free tier in 2026
OPENAI_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a careful medical-information assistant.
Answer the user's question using ONLY the context passages provided below.
If the context does not contain the answer, say so plainly instead of guessing.
Always mention which source(s) (by title) your answer is drawn from.
This is educational information, not a substitute for professional medical advice."""


def get_retriever():
    """Load the persisted Chroma vector store built in step 2."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="medical_guidelines",
    )
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


def build_prompt(question: str, retrieved_docs) -> str:
    """Combine retrieved chunks + the question into a single LLM prompt."""
    context_blocks = []
    for i, doc in enumerate(retrieved_docs, start=1):
        title = doc.metadata.get("title") or "untitled source"
        context_blocks.append(f"[{i}] Source: {title}\n{doc.page_content}")
    context = "\n\n".join(context_blocks)

    return f"""Context passages:
{context}

Question: {question}

Answer the question using only the context above."""


def call_llm(prompt: str) -> str:
    """Send the prompt to the configured cloud LLM API and return the text."""
    if LLM_PROVIDER == "groq":
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

    elif LLM_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def ask(question: str, retriever) -> None:
    retrieved_docs = retriever.invoke(question)

    print("\n--- Retrieved chunks (vector search results) ---")
    for i, doc in enumerate(retrieved_docs, start=1):
        title = doc.metadata.get("title") or "untitled source"
        snippet = doc.page_content[:120].replace("\n", " ")
        print(f"[{i}] {title} :: {snippet}...")

    prompt = build_prompt(question, retrieved_docs)
    answer = call_llm(prompt)

    print("\n--- Answer ---")
    print(answer)


def main() -> None:
    if LLM_PROVIDER == "groq" and not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your "
            "free key from https://console.groq.com/keys"
        )

    retriever = get_retriever()
    print("Medical RAG assistant ready. Type a question, or 'quit' to exit.")
    print("(Try: 'What are the recommendations for treating malaria?')\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue
        ask(question, retriever)


if __name__ == "__main__":
    main()