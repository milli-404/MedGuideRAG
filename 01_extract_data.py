"""
STEP 1 (v2): Build a deliberate, documented corpus.

Changes from the baseline:
  - Only WHO and CDC documents (WikiDoc dropped: it is a wiki, not a guideline body).
  - Documents are selected by TOPIC, with a per-(source, topic) quota,
    so coverage is intentional instead of "the first 300 rows".
  - Writes data/corpus_manifest.csv describing exactly what is in the corpus.

Run:  python 01_extract_data.py
Note: it streams the whole dataset to find topic matches, so it can take
several minutes. Check the dataset card on Hugging Face for licensing
before you add other sources (e.g. nice, cma, pubmed).
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# ---- Configuration ----
SOURCES_TO_KEEP = {"who", "cdc"}
PER_TOPIC_PER_SOURCE = 25      # quota: max docs per (source, topic)
MIN_CHARS = 1000               # skip near-empty / stub documents
TEXT_WINDOW = 3000             # only scan the start of the text for keywords
MIN_BODY_HITS = 3              # keyword hits needed if the title doesn't match

# topic -> regex of keywords (case-insensitive)
TOPICS = {
    "malaria": r"malaria|plasmodium",
    "tuberculosis": r"tuberculosis",
    "hiv": r"\bhiv\b|antiretroviral",
    "hypertension": r"hypertension|blood pressure",
    "diabetes": r"diabetes|diabetic",
    "immunization": r"immuni[sz]ation|vaccin\w*",
    "infection_control": r"hand hygiene|infection prevention|infection control",
    "covid19": r"covid|sars-cov-2",
    "influenza": r"influenza|\bflu\b",
    "maternal_health": r"pregnan\w*|antenatal|postpartum|maternal",
    "child_nutrition": r"malnutrition|breastfeeding|child nutrition|stunting",
    "diarrhoeal_disease": r"cholera|diarrh?oea|diarrhea|oral rehydration",
    "hepatitis": r"hepatitis",
    "antimicrobial_resistance": r"antimicrobial resistance|antibiotic resistance",
    "mental_health": r"depression|mental health|anxiety",
}
PATTERNS = {t: re.compile(p, re.IGNORECASE) for t, p in TOPICS.items()}

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = DATA_DIR / "raw"
MANIFEST_PATH = DATA_DIR / "corpus_manifest.csv"


def rank_topics(title: str, text: str) -> list[str]:
    """Return qualifying topics, best match first."""
    window = text[:TEXT_WINDOW]
    scored = []
    for topic, pat in PATTERNS.items():
        title_hit = bool(pat.search(title))
        body_hits = len(pat.findall(window))
        if title_hit or body_hits >= MIN_BODY_HITS:
            scored.append((5 * title_hit + min(body_hits, 10), topic))
    scored.sort(reverse=True)
    return [t for _, t in scored]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove the old baseline documents so the corpus is exactly what this run builds.
    for old in OUTPUT_DIR.glob("*.json"):
        old.unlink()

    dataset = load_dataset("epfl-llm/guidelines", split="train", streaming=True)

    counts: Counter = Counter()
    seen_titles: set = set()
    manifest_rows = []

    for row in tqdm(dataset, desc="Scanning dataset"):
        source = row["source"]
        if source not in SOURCES_TO_KEEP:
            continue
        text = row["clean_text"] or ""
        if len(text) < MIN_CHARS:
            continue

        title = row["title"] or f"{source}-{row['id'][:8]}"
        key = (source, title.strip().lower())
        if key in seen_titles:
            continue  # skip duplicate titles from the same source

        chosen = None
        for topic in rank_topics(title, text):
            if counts[(source, topic)] < PER_TOPIC_PER_SOURCE:
                chosen = topic
                break
        if chosen is None:
            continue

        seen_titles.add(key)
        counts[(source, chosen)] += 1

        doc = {
            "id": row["id"],
            "source": source,
            "title": title,
            "url": row["url"] or "",
            "topic": chosen,
            "text": text,
        }
        with open(OUTPUT_DIR / f"{row['id']}.json", "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        manifest_rows.append(
            {
                "id": row["id"],
                "source": source,
                "title": title,
                "topic": chosen,
                "n_chars": len(text),
                "url": doc["url"],
            }
        )

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "source", "title", "topic", "n_chars", "url"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nSaved {len(manifest_rows)} documents to {OUTPUT_DIR}")
    print(f"Manifest written to {MANIFEST_PATH}\n")
    print(f"{'source':8} {'topic':26} docs")
    for source in sorted(SOURCES_TO_KEEP):
        for topic in TOPICS:
            n = counts[(source, topic)]
            flag = "  <-- under quota" if n < PER_TOPIC_PER_SOURCE else ""
            print(f"{source:8} {topic:26} {n:4}{flag}")


if __name__ == "__main__":
    main()