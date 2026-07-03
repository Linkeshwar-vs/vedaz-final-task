"""
Run : python scripts/preprocessing/deep_audit.py data/raw/dataset.json
"""



import argparse
import json
import hashlib
import os
import re
from pathlib import Path
from collections import Counter, defaultdict

Path("reports").mkdir(parents=True, exist_ok=True)

REPORT_FILE = "reports/Dataset_Audit_Report_Part2.txt"

STOPWORDS = {
    "the","a","an","is","are","was","were","to","of","and","or","in","on","for",
    "hai","hain","ki","ka","ke","ko","mein","main","se","par","aur","ya","ye",
    "woh","ho","kar","do","kya","you","your","i","we","it","this","that"
}

def estimate_tokens(text):
    return max(1, int(len(text.split()) * 1.33))

def load_dataset(path):
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
    except:
        pass

    decoder = json.JSONDecoder()
    idx = 0
    data = []
    while idx < len(raw):
        while idx < len(raw) and raw[idx] in ", \n\r\t":
            idx += 1
        if idx >= len(raw):
            break
        try:
            obj, end = decoder.raw_decode(raw, idx)
            data.append(obj)
            idx = end
        except:
            idx += 1
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()

    data = load_dataset(args.dataset)

    conv_hash_map = defaultdict(list)
    assistant_hash_map = defaultdict(list)
    user_hash_map = defaultdict(list)
    system_counter = Counter()
    vocab = Counter()

    longest = []
    longest_assistant = []
    shortest_assistant = []

    total_words = 0

    for idx, conv in enumerate(data, start=1):
        msgs = conv.get("messages", [])
        serial = json.dumps(msgs, ensure_ascii=False, sort_keys=True)
        conv_hash_map[hashlib.md5(serial.encode()).hexdigest()].append(idx)

        chars = 0
        toks = 0

        for msg in msgs:
            role = msg.get("role", "")
            txt = msg.get("content", "")

            chars += len(txt)
            toks += estimate_tokens(txt)

            if role == "assistant":
                h = hashlib.md5(txt.strip().encode()).hexdigest()
                assistant_hash_map[h].append(idx)
                longest_assistant.append((len(txt), idx, txt[:120]))
                shortest_assistant.append((len(txt), idx, txt[:120]))

            elif role == "user":
                h = hashlib.md5(txt.strip().encode()).hexdigest()
                user_hash_map[h].append(idx)

            elif role == "system":
                system_counter[txt.strip()] += 1

            words = re.findall(r"\b[\w']+\b", txt.lower())
            for w in words:
                if w not in STOPWORDS and len(w) > 2:
                    vocab[w] += 1
                    total_words += 1

        longest.append((chars, toks, len(msgs), idx))

    longest.sort(reverse=True)
    longest_assistant.sort(reverse=True)
    shortest_assistant.sort()

    duplicate_convs = {k:v for k,v in conv_hash_map.items() if len(v)>1}
    duplicate_assistant = {k:v for k,v in assistant_hash_map.items() if len(v)>1}
    duplicate_user = {k:v for k,v in user_hash_map.items() if len(v)>1}

    unique_system = len(system_counter)
    repeated_system = sum(1 for c in system_counter.values() if c>1)

    lexical_div = (len(vocab)/total_words) if total_words else 0

    avg_tokens = sum(t for _,t,_,_ in longest)/len(longest) if longest else 0

    if avg_tokens <= 256:
        seq = 256
    elif avg_tokens <= 512:
        seq = 512
    else:
        seq = 1024

    if avg_tokens <= 200:
        epochs = 8
    elif avg_tokens <= 400:
        epochs = 5
    else:
        epochs = 3

    with open(REPORT_FILE,"w",encoding="utf-8") as f:
        f.write("="*70+"\n")
        f.write("VEDAZ AI - PHASE 1.1 DEEP AUDIT REPORT\n")
        f.write("="*70+"\n\n")

        f.write("1. DUPLICATE CONVERSATIONS\n")
        f.write("-"*40+"\n")
        if duplicate_convs:
            for v in duplicate_convs.values():
                f.write(f"{v}\n")
        else:
            f.write("None\n")

        f.write("\n2. DUPLICATE ASSISTANT RESPONSES\n")
        f.write("-"*40+"\n")
        for v in duplicate_assistant.values():
            f.write(f"Used in conversations: {v}\n")

        f.write("\n3. DUPLICATE USER PROMPTS\n")
        f.write("-"*40+"\n")
        for v in duplicate_user.values():
            f.write(f"Used in conversations: {v}\n")

        f.write("\n4. SYSTEM PROMPTS\n")
        f.write("-"*40+"\n")
        f.write(f"Unique : {unique_system}\n")
        f.write(f"Repeated : {repeated_system}\n")
        f.write("Top repeated:\n")
        for txt,c in system_counter.most_common(10):
            f.write(f"{c}x : {txt[:90]}...\n")

        f.write("\n5. VOCABULARY\n")
        f.write("-"*40+"\n")
        f.write(f"Unique Words : {len(vocab)}\n")
        f.write(f"Total Words : {total_words}\n")
        f.write(f"Lexical Diversity : {lexical_div:.4f}\n")

        f.write("\nTop 20 Words\n")
        for w,c in vocab.most_common(20):
            f.write(f"{w}: {c}\n")

        f.write("\n6. LONGEST CONVERSATIONS\n")
        f.write("-"*40+"\n")
        for chars,tok,msgs,idx in longest[:10]:
            f.write(f"Conversation {idx} | Messages={msgs} | Chars={chars} | Tokens≈{tok}\n")

        f.write("\n7. LONGEST ASSISTANT RESPONSES\n")
        f.write("-"*40+"\n")
        for l,idx,preview in longest_assistant[:10]:
            f.write(f"Conversation {idx} | {l} chars | {preview}\n")

        f.write("\n8. SHORTEST ASSISTANT RESPONSES\n")
        f.write("-"*40+"\n")
        for l,idx,preview in shortest_assistant[:10]:
            f.write(f"Conversation {idx} | {l} chars | {preview}\n")

        f.write("\n9. TRAINING RECOMMENDATIONS\n")
        f.write("-"*40+"\n")
        f.write(f"Training Examples : {len(data)}\n")
        f.write(f"Average Tokens / Example : {avg_tokens:.2f}\n")
        f.write(f"Recommended Max Sequence Length : {seq}\n")
        f.write("Recommended Batch Size : 1 (RTX 3050 4GB)\n")
        f.write(f"Recommended Epochs : {epochs}\n")
        f.write("Recommended Method : QLoRA\n")

        f.write("\n10. RECOMMENDATIONS\n")
        f.write("-"*40+"\n")
        f.write("• Review duplicate conversations before removing them.\n")
        f.write("• Keep multilingual balance (English/Hindi/Hinglish).\n")
        f.write("• Expand dataset before production training.\n")
        f.write("• Dataset is best suited for behavioural alignment rather than knowledge acquisition.\n")

    print(f"Generated {REPORT_FILE}")

if __name__ == "__main__":
    main()
