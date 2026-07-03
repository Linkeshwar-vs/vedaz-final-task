
"""
Run : 

python scripts/preprocessing/verify_dataset.py ^
data/processed/cleaned_dataset.json ^
data/processed/training_dataset.jsonl ^
data/processed/validation_dataset.jsonl

"""

import argparse
import json
import hashlib
from pathlib import Path

REPORT_FILE = "reports/Dataset_Verification_Report.txt"
SUMMARY_FILE = "reports/verification_summary.json"

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_jsonl(path):
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def conv_hash(conv):
    return hashlib.md5(
        json.dumps(conv["messages"],sort_keys=True,ensure_ascii=False).encode()
    ).hexdigest()

def estimate_tokens(text):
    return max(1,int(len(text.split())*1.33))

def verify_structure(conv):
    msgs=conv.get("messages")
    if not isinstance(msgs,list) or not msgs:
        return False
    if msgs[0].get("role")!="system":
        return False
    expect="user"
    for m in msgs[1:]:
        if "role" not in m or "content" not in m:
            return False
        if not m["content"].strip():
            return False
        if m["role"]!=expect:
            return False
        expect="assistant" if expect=="user" else "user"
    return msgs[-1]["role"]=="assistant"

def main():
    ap=argparse.ArgumentParser()

    Path("reports").mkdir(parents=True, exist_ok=True)

    ap.add_argument("cleaned")
    ap.add_argument("train")
    ap.add_argument("val")
    args=ap.parse_args()

    cleaned=load_json(args.cleaned)
    train=load_jsonl(args.train)
    val=load_jsonl(args.val)

    summary={}
    summary["cleaned_examples"]=len(cleaned)
    summary["train_examples"]=len(train)
    summary["validation_examples"]=len(val)

    # IDs
    ids=[c.get("id") for c in cleaned]
    summary["unique_ids"]=len(ids)==len(set(ids))

    # Structure
    bad=[]
    for c in cleaned:
        if not verify_structure(c):
            bad.append(c.get("id"))
    summary["structure_pass"]=len(bad)==0
    summary["invalid_conversations"]=bad

    # Overlap
    train_hash={conv_hash(c) for c in train}
    val_hash={conv_hash(c) for c in val}
    overlap=train_hash & val_hash
    summary["overlap_count"]=len(overlap)

    # Duplicate check
    all_hashes=[conv_hash(c) for c in cleaned]
    summary["duplicate_count"]=len(all_hashes)-len(set(all_hashes))

    # Token stats
    def token_stats(rows):
        vals=[]
        for c in rows:
            vals.append(sum(estimate_tokens(m["content"]) for m in c["messages"]))
        return {
            "count":len(vals),
            "avg":round(sum(vals)/len(vals),2) if vals else 0,
            "min":min(vals) if vals else 0,
            "max":max(vals) if vals else 0
        }

    summary["train_tokens"]=token_stats(train)
    summary["validation_tokens"]=token_stats(val)

    summary["hf_compatible"]=all("messages" in c and isinstance(c["messages"],list) for c in train+val)

    passed = (
        summary["unique_ids"] and
        summary["structure_pass"] and
        summary["duplicate_count"]==0 and
        summary["overlap_count"]==0 and
        summary["hf_compatible"] and
        (len(train)+len(val)==len(cleaned))
    )

    summary["ready_for_training"]=passed

    with open(SUMMARY_FILE,"w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2,ensure_ascii=False)

    with open(REPORT_FILE,"w",encoding="utf-8") as f:
        f.write("="*70+"\n")
        f.write("VEDAZ AI - PHASE 2 VERIFICATION REPORT\n")
        f.write("="*70+"\n\n")
        f.write(f"JSON Integrity ............. PASS\n")
        f.write(f"Conversation Structure ..... {'PASS' if summary['structure_pass'] else 'FAIL'}\n")
        f.write(f"Unique IDs ................ {'PASS' if summary['unique_ids'] else 'FAIL'}\n")
        f.write(f"Duplicates ................ {'PASS' if summary['duplicate_count']==0 else 'FAIL'} ({summary['duplicate_count']})\n")
        f.write(f"Train/Val Overlap ......... {'PASS' if summary['overlap_count']==0 else 'FAIL'} ({summary['overlap_count']})\n")
        f.write(f"HF Compatibility .......... {'PASS' if summary['hf_compatible'] else 'FAIL'}\n")
        f.write(f"Train + Val == Cleaned .... {'PASS' if (len(train)+len(val)==len(cleaned)) else 'FAIL'}\n\n")
        f.write("Training Token Statistics\n")
        f.write("-------------------------\n")
        for k,v in summary["train_tokens"].items():
            f.write(f"{k}: {v}\n")
        f.write("\nValidation Token Statistics\n")
        f.write("---------------------------\n")
        for k,v in summary["validation_tokens"].items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
        if passed:
            f.write("STATUS: READY FOR TRAINING\n")
        else:
            f.write("STATUS: NOT READY FOR TRAINING\n")

    print("Verification complete.")
    print("Generated:", REPORT_FILE)
    print("Generated:", SUMMARY_FILE)

if __name__=="__main__":
    main()
