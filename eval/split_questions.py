"""
PHASE 2c: Turn your reviewed questions into the final evaluation set.

Input : eval/questions_reviewed.json  (your hand-reviewed file; see below)
Output: eval/questions.jsonl          (one question per line, with a dev/test split)

Reviewed file format: a JSON list. Keep entries with "status": "ok".
Answerable entries come from the draft script. Unanswerable entries you
write yourself, in this shape:
  {
    "qid": "u001",
    "type": "unanswerable",
    "status": "ok",
    "rewritten": true,
    "question": "...",
    "unanswerable_kind": "out_of_scope" | "near_miss",
    "topic": "malaria"        # the nearest topic, or "none" for out_of_scope
  }

Run from the project root:  python eval/split_questions.py
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "eval" / "questions_reviewed.json"
OUT_PATH = ROOT / "eval" / "questions.jsonl"
SEED = 42


def validate(q: dict) -> list[str]:
    problems = []
    if not q.get("question", "").strip():
        problems.append("empty question")
    if q.get("type") == "answerable":
        for field in ("reference_answer", "gold_doc_id", "gold_chunk_id"):
            if not q.get(field):
                problems.append(f"missing {field}")
    elif q.get("type") == "unanswerable":
        if q.get("unanswerable_kind") not in {"out_of_scope", "near_miss"}:
            problems.append("unanswerable_kind must be out_of_scope or near_miss")
    else:
        problems.append("type must be answerable or unanswerable")
    return problems


def main() -> None:
    with open(IN_PATH, encoding="utf-8") as f:
        items = json.load(f)

    kept, bad = [], 0
    for q in items:
        if q.get("status") != "ok":
            continue
        problems = validate(q)
        if problems:
            bad += 1
            print(f"SKIPPED {q.get('qid')}: {', '.join(problems)}")
            continue
        kept.append(q)

    # Stratified 50/50 split: group by (type, topic), shuffle, alternate dev/test.
    random.seed(SEED)
    groups = defaultdict(list)
    for q in kept:
        groups[(q["type"], q.get("topic", "none"))].append(q)

    flip = 0
    for key in sorted(groups):
        random.shuffle(groups[key])
        for q in groups[key]:
            q["split"] = "dev" if flip % 2 == 0 else "test"
            flip += 1

    final = sorted(kept, key=lambda q: q["qid"])
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for q in final:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(final)} questions to {OUT_PATH} ({bad} skipped as invalid)")
    print("By type/split:", dict(Counter((q["type"], q["split"]) for q in final)))
    rewritten = sum(1 for q in final if q.get("rewritten"))
    print(f"Rewritten by hand: {rewritten} of {len(final)}")


if __name__ == "__main__":
    main()