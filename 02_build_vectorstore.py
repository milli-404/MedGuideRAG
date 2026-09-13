"""
STEP 2: Build the RAG knowledge base (chunking + embeddings + vector DB).

What this script does:
  1. Loads the JSON files saved by 01_extract_data.py.
  2. Splits each document into overlapping ~500-token chunks. We chunk
     because embedding models and LLMs work better on short, focused
     passages than on giant 20-page documents, and because retrieval
     needs to find the *specific* paragraph that answers a question.
  3. Converts each chunk into a vector (an "embedding") using a free,
     open-source sentence-transformers model that runs on your own CPU
     -- no API key or cost required for this step.
  4. Stores all vectors in a Chroma vector database on disk, so step 3
     can search over them instantly without recomputing anything.

Beginner notes on the key concepts:
  - "Embedding" = a list of a few hundred numbers that represents the
    *meaning* of a piece of text. Similar meanings -> similar numbers.
  - "Vector search" = given a question's embedding, find the stored
    chunks whose embeddings are numerically closest to it (cosine
    similarity). That's how the system finds relevant medical text
    without keyword matching.
  - Chroma is a lightweight, file-based vector database -- perfect for
    learning and small/medium projects. Swap it for Pinecone or a
    hosted vector DB later if you need to scale to millions of chunks.
"""

import json
from pathlib import Path

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

RAW_DIR = Path(__file__).parent / "data" / "raw"
PERSIST_DIR = Path(__file__).parent / "vectorstore"

# A small, fast, free embedding model. 384-dimensional vectors, runs on CPU.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800        # characters per chunk (roughly 150-200 words)
CHUNK_OVERLAP = 120     # overlap so we don't cut a sentence in half between chunks


def load_documents() -> list[Document]:
    """Read every JSON file in data/raw/ into a LangChain Document."""
    documents = []
    for path in RAW_DIR.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            row = json.load(f)
        documents.append(
            Document(
                page_content=row["text"],
                metadata={
                    "title": row["title"],
                    "source": row["source"],
                    "url": row["url"],
                    "doc_id": row["id"],
                },
            )
        )
    return documents


def main() -> None:
    print("Loading raw documents...")
    documents = load_documents()
    if not documents:
        raise SystemExit(
            "No documents found in data/raw/. Run 01_extract_data.py first."
        )
    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try to split on paragraph/sentence boundaries before falling
        # back to raw character cuts.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents.")

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' "
          "(downloads once, then runs locally)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Embedding chunks and writing them to the Chroma vector store. "
          "This is the slowest step -- it's doing real ML inference on "
          "every chunk.")

    # Chroma rejects a single add() call above a few thousand items, so we
    # create an empty collection first, then insert the chunks in batches.
    vectorstore = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name="medical_guidelines",
    )

    BATCH_SIZE = 500
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding batches"):
        batch = chunks[i : i + BATCH_SIZE]
        vectorstore.add_documents(batch)

    print(f"\nDone. Vector store persisted to {PERSIST_DIR}")
    print(f"Total vectors stored: {vectorstore._collection.count()}")


if __name__ == "__main__":
    main()