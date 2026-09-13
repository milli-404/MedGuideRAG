"""
STEP 1: Extract a healthcare dataset.

What this script does:
  1. Downloads the "epfl-llm/guidelines" dataset from Hugging Face Hub.
     This dataset contains ~38,000 real clinical practice guidelines from
     sources like WHO, CDC, NICE, CMA, and PubMed.
  2. Filters it down to a manageable subset (so embedding doesn't take
     hours on a laptop) and picks a few medical sources/topics.
  3. Saves each guideline as a small JSON file on disk, which later steps
     will read in as LangChain "documents".

Beginner notes:
  - `datasets.load_dataset()` streams data straight from the Hugging Face
    Hub -- no manual downloading or unzipping required.
  - We save to plain files in data/raw/ instead of keeping everything in
    memory, so you can inspect the raw text yourself, and so step 2 can
    be re-run without re-downloading anything.
"""

import json
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# ---- Configuration you can tweak ----
# Only keep guidelines from these sources (see dataset card for the full list:
# cco, cdc, cma, icrc, nice, pubmed, spor, who, wikidoc)
SOURCES_TO_KEEP = {"who", "cdc", "wikidoc"}

# Cap the number of documents so this stays fast and cheap to embed.
# Increase this later once the pipeline works end-to-end.
MAX_DOCUMENTS = 300

OUTPUT_DIR = Path(__file__).parent / "data" / "raw"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset metadata from Hugging Face (this streams, "
          "it does not pull all 38k rows into memory)...")
    # streaming=True means rows are fetched one at a time instead of
    # downloading the entire ~1GB dataset up front.
    dataset = load_dataset(
        "epfl-llm/guidelines", split="train", streaming=True
    )

    saved = 0
    with tqdm(total=MAX_DOCUMENTS, desc="Saving guideline documents") as pbar:
        for row in dataset:
            if row["source"] not in SOURCES_TO_KEEP:
                continue
            if not row["clean_text"] or len(row["clean_text"]) < 200:
                continue  # skip near-empty rows

            doc = {
                "id": row["id"],
                "source": row["source"],
                "title": row["title"] or f"{row['source']}-{row['id'][:8]}",
                "url": row["url"],
                "text": row["clean_text"],
            }

            out_path = OUTPUT_DIR / f"{row['id']}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)

            saved += 1
            pbar.update(1)
            if saved >= MAX_DOCUMENTS:
                break

    print(f"\nDone. Saved {saved} guideline documents to {OUTPUT_DIR}")
    print("Open one of the .json files to see what raw medical guideline "
          "data looks like before we chunk and embed it.")


if __name__ == "__main__":
    main()
