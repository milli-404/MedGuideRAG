# Medical guidelines RAG system

A from-scratch Retrieval-Augmented Generation (RAG) project over real
clinical practice guidelines. Built to touch every core ML-engineering
skill: dataset extraction, embeddings, vector search, and cloud LLM APIs.

**Educational project only.** This is not a medical device and must never
be used for real clinical decisions.

## What you're building

```
Offline (run once):
  Medical guidelines --> chunk --> embed --> store in vector DB

Online (run every question):
  Your question --> embed --> vector search --> top-k chunks
                                                       |
                                                       v
                                     LLM API (Groq/OpenAI) --> grounded answer
```

This pattern (RAG) is the standard way to make an LLM answer questions
about a specific body of knowledge it wasn't trained on, without
fine-tuning a model.

## Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys) (takes 1 minute,
  no credit card) -- this is your cloud LLM.
- ~2GB free disk space for the embedding model and vector store.

## Setup

```bash
cd medical-rag
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# open .env and paste your GROQ_API_KEY in
```

## Run it, step by step

### Step 1 — Extract the dataset

```bash
python 01_extract_data.py
```

This downloads real clinical guidelines from the `epfl-llm/guidelines`
dataset on Hugging Face (the same corpus used to train the Meditron
medical LLM) and saves ~300 of them as JSON files in `data/raw/`.

Open one of the files in `data/raw/` afterward -- that's what raw
healthcare data looks like before any ML happens to it.

**Concept:** this is the "extract" step of any ML pipeline. In a real
job you might pull this from an internal database, an S3 bucket of
PDFs, or a partner's API instead of a public dataset -- the rest of the
pipeline doesn't care where the text came from.

### Step 2 — Chunk, embed, and index

```bash
python 02_build_vectorstore.py
```

This is where the actual ML happens:

1. **Chunking** — Documents are split into ~800-character overlapping
   pieces. LLMs and embedding models work on limited-size windows, and
   retrieval needs paragraph-level granularity, not whole-document
   granularity, to find the exact relevant passage.
2. **Embedding** — Each chunk is converted into a 384-number vector using
   `sentence-transformers/all-MiniLM-L6-v2`, a small open-source model
   that runs on your CPU, no API key needed. Texts with similar meaning
   end up with numerically similar vectors.
3. **Vector storage** — All vectors go into **Chroma**, a lightweight
   vector database that saves itself to disk in `vectorstore/`. Later,
   given a new vector (your question), Chroma can instantly find the
   closest stored vectors using cosine similarity — this is "vector
   search."

This step can take a few minutes the first time (it downloads the
embedding model once, then runs local inference on every chunk).

### Step 3 — Ask questions (retrieval + generation)

```bash
python 03_rag_chat.py
```

For every question you type, the script:

1. Embeds your question with the same embedding model from step 2.
2. Runs a vector search against Chroma to pull the top 4 most relevant
   chunks — this is the **R** (retrieval) in RAG.
3. Builds a prompt that says "here is context, here is the question,
   answer using only the context."
4. Sends that prompt to **Groq's cloud API**, which serves an
   open-source LLM (Llama 3.1) — this is the **G** (generation) and the
   cloud/API-integration part of the project.
5. Prints the answer along with which sources it used.

Try asking things like:
- "What are the recommendations for treating malaria?"
- "What does the CDC guidance say about hand hygiene?"

If you get a weak answer, it's usually because the small 300-document
sample doesn't cover that topic — increase `MAX_DOCUMENTS` in
`01_extract_data.py` and re-run steps 1–2.

### Optional: web UI

```bash
streamlit run app.py
```

Same logic as step 3, wrapped in a browser-based chat interface.

## Where each required skill lives

| Requirement | Where |
|---|---|
| Extract a healthcare dataset | `01_extract_data.py` (Hugging Face `datasets`) |
| RAG over documents with LangChain | `02_build_vectorstore.py`, `03_rag_chat.py` |
| Vector DB (Chroma/FAISS) | `Chroma` in `02_build_vectorstore.py` |
| Embeddings | `HuggingFaceEmbeddings` (sentence-transformers, local) |
| Open-source or API-based LLM | Groq API serving Llama 3.1 (swap to OpenAI in `03_rag_chat.py`) |
| Cloud/API integration | `groq`/`openai` client calls in `call_llm()` |

## Natural next steps once this works

- **Swap the vector DB**: replace `Chroma` with `Pinecone` (hosted,
  scales to millions of vectors) — only `02_build_vectorstore.py` and
  the `get_retriever()` function in `03_rag_chat.py` change.
- **Add re-ranking**: after vector search returns top-20, use a
  cross-encoder to re-rank down to the best 4 — improves precision.
- **Add evaluation**: build a small set of Q&A pairs and measure whether
  retrieved chunks actually contain the answer (retrieval recall).
- **Swap FAISS in**: FAISS is Meta's library for pure similarity search
  (no metadata filtering, but very fast) — good to learn as an
  alternative to Chroma for large-scale/production use.
- **Guardrails**: add a check that refuses to answer if retrieval
  confidence is too low, instead of letting the LLM guess.
# MedGuideRAG
