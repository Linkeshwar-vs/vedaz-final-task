"""
Run : python scripts/preprocessing/prepare_dataset.py data/raw/dataset.json
"""


import argparse
import json
import hashlib
import random
import re
from pathlib import Path
from collections import Counter

RANDOM_SEED = 42
TRAIN_RATIO = 0.90

def estimate_tokens(text):
    return max(1, int(len(text.split()) * 1.33))

def normalize(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def load_dataset(path):
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    decoder = json.JSONDecoder()
    idx = 0
    out = []
    while idx < len(raw):
        while idx < len(raw) and raw[idx] in " \n\r\t,":
            idx += 1
        if idx >= len(raw):
            break
        try:
            obj, end = decoder.raw_decode(raw, idx)
            out.append(obj)
            idx = end
        except Exception:
            idx += 1
    return out

def validate(messages):
    if not messages:
        return False
    if messages[0].get("role") != "system":
        return False
    expect = "user"
    for i, m in enumerate(messages[1:], start=1):
        role = m.get("role")
        if role not in ("system","user","assistant"):
            return False
        if role != expect:
            return False
        expect = "assistant" if expect=="user" else "user"
        if not m.get("content","").strip():
            return False
    return True

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    args = ap.parse_args()

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)

    raw = load_dataset(args.dataset)

    cleaned = []
    seen = set()

    stats = {
        "loaded": len(raw),
        "duplicates_removed": 0,
        "invalid_removed": 0,
        "avg_tokens": 0,
        "max_tokens": 0,
        "min_tokens": 0,
    }

    token_counts = []

    for conv in raw:
        msgs = conv.get("messages", [])
        if not validate(msgs):
            stats["invalid_removed"] += 1
            continue

        norm_msgs = []
        for m in msgs:
            norm_msgs.append({
                "role": m["role"],
                "content": normalize(m["content"])
            })

        signature = hashlib.md5(
            json.dumps(norm_msgs, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

        if signature in seen:
            stats["duplicates_removed"] += 1
            continue

        seen.add(signature)

        total_tok = sum(estimate_tokens(x["content"]) for x in norm_msgs)
        token_counts.append(total_tok)

        item = {
            "id": f"conv_{len(cleaned)+1:04d}",
            "messages": norm_msgs
        }

        if "tags" in conv:
            item["tags"] = conv["tags"]

        cleaned.append(item)

    if token_counts:
        stats["avg_tokens"] = round(sum(token_counts)/len(token_counts),2)
        stats["max_tokens"] = max(token_counts)
        stats["min_tokens"] = min(token_counts)

    stats["final_examples"] = len(cleaned)

    # simple stratification by first tag where available
    grouped = {}
    untagged = []
    for c in cleaned:
        tags = c.get("tags", [])
        if tags:
            grouped.setdefault(tags[0], []).append(c)
        else:
            untagged.append(c)

    train=[]
    val=[]

    for lst in grouped.values():
        random.shuffle(lst)
        split=max(1,int(len(lst)*TRAIN_RATIO))
        train.extend(lst[:split])
        val.extend(lst[split:])

    random.shuffle(untagged)
    split=max(1,int(len(untagged)*TRAIN_RATIO)) if untagged else 0
    train.extend(untagged[:split])
    val.extend(untagged[split:])

    random.shuffle(train)
    random.shuffle(val)

    Path("data/processed/cleaned_dataset.json").write_text(
        json.dumps(cleaned, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    write_jsonl("data/processed/training_dataset.jsonl", train)
    write_jsonl("data/processed/validation_dataset.jsonl", val)

    Path("reports/dataset_statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    report = f"""VEDAZ AI - CLEANING REPORT
========================================

Loaded Conversations      : {stats['loaded']}
Duplicates Removed        : {stats['duplicates_removed']}
Invalid Removed           : {stats['invalid_removed']}
Final Examples            : {stats['final_examples']}

Training Examples         : {len(train)}
Validation Examples       : {len(val)}

Average Tokens            : {stats['avg_tokens']}
Maximum Tokens            : {stats['max_tokens']}
Minimum Tokens            : {stats['min_tokens']}

Files Generated
---------------
data/processed/cleaned_dataset.json
data/processed/training_dataset.jsonl
data/processed/validation_dataset.jsonl
reports/dataset_statistics.json
reports/Dataset_Cleaning_Report.txt

Dataset Status
--------------
READY FOR QWEN SFT
"""

    Path("reports/Dataset_Cleaning_Report.txt").write_text(report, encoding="utf-8")

    print("Completed successfully.")

if __name__ == "__main__":
    main()
