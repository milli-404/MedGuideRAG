"""
STEP 2 (v2): Chunk, embed, and index the corpus built by 01_extract_data.py.

Changes from the baseline:
  - Every chunk gets a stable chunk_id ("<doc_id>_<index>") and a chunk_index,
    so the evaluation can check whether the RIGHT chunk was retrieved.
  - The document's topic is stored in each chunk's metadata.
  - The old vector store is deleted before building, so re-running never
    leaves stale or duplicated chunks behind.

Run:  python 02_build_vectorstore.py
"""

import json
import shutil
from pathlib import Path

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

RAW_DIR = Path(__file__).parent / "data" / "raw"
PERSIST_DIR = Path(__file__).parent / "vectorstore"
COLLECTION_NAME = "medical_guidelines"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 120     # overlap so sentences aren't cut between chunks
BATCH_SIZE = 500

REBUILD = True          # delete the existing vector store before building


def load_documents() -> list[Document]:
    """Read every JSON file in data/raw/ into a LangChain Document."""
    documents = []
    for path in sorted(RAW_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            row = json.load(f)
        documents.append(
            Document(
                page_content=row["text"],
                metadata={
                    "title": row["title"],
                    "source": row["source"],
                    "url": row.get("url") or "",
                    "doc_id": row["id"],
                    "topic": row.get("topic", "unknown"),
                },
            )
        )
    return documents


def add_chunk_ids(chunks: list[Document]) -> None:
    """Give each chunk a stable id and its position within its document."""
    counters: dict[str, int] = {}
    for chunk in chunks:
        doc_id = chunk.metadata["doc_id"]
        idx = counters.get(doc_id, 0)
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"] = f"{doc_id}_{idx}"
        counters[doc_id] = idx + 1


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
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    add_chunk_ids(chunks)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents.")

    if REBUILD and PERSIST_DIR.exists():
        print(f"Removing old vector store at {PERSIST_DIR}...")
        shutil.rmtree(PERSIST_DIR)

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' "
          "(downloads once, then runs locally)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vectorstore = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    # Insert in batches: Chroma rejects very large single add() calls.
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding batches"):
        batch = chunks[i : i + BATCH_SIZE]
        vectorstore.add_documents(
            batch, ids=[c.metadata["chunk_id"] for c in batch]
        )

    print(f"\nDone. Vector store persisted to {PERSIST_DIR}")
    print(f"Total vectors stored: {vectorstore._collection.count()}")


if __name__ == "__main__":
    main()