"""
Run : python scripts/preprocessing/dataset_audit.py data/raw/dataset.json
"""


import argparse, json, os, re, hashlib
from pathlib import Path
from collections import Counter
from statistics import mean, median

Path("reports").mkdir(parents=True, exist_ok=True)

REPORT_FILE = "reports/Dataset_Audit_Report_Part1.txt"
SUMMARY_FILE = "reports/audit_summary.json"

def estimate_tokens(text:str)->int:
    return max(1,int(len(text.split())*1.33))

class Loader:
    def __init__(self,path):
        self.path=path
        self.format="Unknown"
        self.recovered=0

    def load(self):
        raw=Path(self.path).read_text(encoding="utf-8",errors="ignore")
        try:
            obj=json.loads(raw)
            if isinstance(obj,list):
                self.format="JSON Array"
                return obj
        except Exception:
            pass
        try:
            rows=[]
            for line in raw.splitlines():
                line=line.strip()
                if line:
                    rows.append(json.loads(line))
            self.format="JSONL"
            return rows
        except Exception:
            pass
        dec=json.JSONDecoder()
        idx=0
        out=[]
        while idx<len(raw):
            while idx<len(raw) and raw[idx] in ", \n\r\t":
                idx+=1
            if idx>=len(raw):
                break
            try:
                obj,end=dec.raw_decode(raw,idx)
                out.append(obj)
                idx=end
            except Exception:
                idx+=1
        self.format="Recovered Concatenated JSON"
        self.recovered=len(out)
        return out

class Auditor:
    def __init__(self,data,loader):
        self.data=data
        self.loader=loader
        self.summary={}

    def run(self):
        print("[1/10] Auditing dataset...")
        convs=len(self.data)
        msgs=0
        roles=Counter()
        depths=[]
        lang=Counter()
        tags=Counter()
        dup_conv=0
        dup_reply=0
        conv_hash=set()
        reply_hash=set()
        invalid_roles=0
        empty=0
        role_errors=0
        tokens=[]
        assistant_chars=[]
        user_chars=[]
        system_chars=[]
        safety=Counter()
        safety_map={
            "Medical":["doctor","hospital","cancer","pregnancy","biopsy"],
            "Mental Health":["suicide","self-harm","helpline","counsellor","counselor"],
            "Finance":["lottery","crypto","financial","advisor"],
            "Legal":["lawyer","court","legal","property"],
            "Relationships":["marriage","breakup","relationship","divorce"],
            "Death/Fear":["death","accident"],
            "Visa":["visa","canada"],
            "Black Magic":["black magic","kala jadu"]
        }
        dev=re.compile(r'[\u0900-\u097F]')
        eng=re.compile(r'[A-Za-z]')
        for conv in self.data:
            m=conv.get("messages",[])
            msgs+=len(m)
            depths.append(len(m))
            serial=json.dumps(m,ensure_ascii=False,sort_keys=True)
            h=hashlib.md5(serial.encode()).hexdigest()
            if h in conv_hash: dup_conv+=1
            conv_hash.add(h)
            prev=None
            txtall=[]
            for msg in m:
                role=msg.get("role","")
                content=msg.get("content","")
                txtall.append(content.lower())
                if not content.strip(): empty+=1
                if role not in ("system","user","assistant"):
                    invalid_roles+=1
                if prev==role:
                    role_errors+=1
                prev=role
                roles[role]+=1
                tk=estimate_tokens(content)
                tokens.append(tk)
                if role=="assistant":
                    assistant_chars.append(len(content))
                    rh=hashlib.md5(content.strip().encode()).hexdigest()
                    if rh in reply_hash: dup_reply+=1
                    reply_hash.add(rh)
                elif role=="user":
                    user_chars.append(len(content))
                elif role=="system":
                    system_chars.append(len(content))
            joined=" ".join(txtall)
            he=bool(eng.search(joined))
            hh=bool(dev.search(joined))
            if he and hh: lang["Mixed"]+=1
            elif hh: lang["Hindi"]+=1
            elif he: lang["English"]+=1
            else: lang["Unknown"]+=1
            for t in conv.get("tags",[]):
                tags[t]+=1
            for cat,keys in safety_map.items():
                if any(k in joined for k in keys):
                    safety[cat]+=1
        self.summary={
            "file":os.path.basename(self.loader.path),
            "format":self.loader.format,
            "recovered_records":self.loader.recovered,
            "dataset_size_mb":round(os.path.getsize(self.loader.path)/1024/1024,3),
            "total_conversations":convs,
            "total_messages":msgs,
            "role_counts":dict(roles),
            "average_depth":round(mean(depths),2) if depths else 0,
            "median_depth":median(depths) if depths else 0,
            "depth_distribution":dict(Counter(depths)),
            "languages":dict(lang),
            "tags":dict(tags),
            "duplicates":{"conversations":dup_conv,"assistant_replies":dup_reply},
            "validation":{"invalid_roles":invalid_roles,"empty_messages":empty,"role_order_errors":role_errors},
            "token_stats":{
                "total":sum(tokens),
                "average":round(mean(tokens),2) if tokens else 0,
                "max":max(tokens) if tokens else 0,
                "min":min(tokens) if tokens else 0
            },
            "message_lengths":{
                "assistant_avg":round(mean(assistant_chars),2) if assistant_chars else 0,
                "user_avg":round(mean(user_chars),2) if user_chars else 0,
                "system_avg":round(mean(system_chars),2) if system_chars else 0
            },
            "safety_coverage":dict(safety)
        }
        score=100-invalid_roles*2-empty-role_errors*0.5-dup_conv
        score=max(0,round(score,2))
        self.summary["health_score"]=score
        self.summary["status"]="READY FOR PHASE 2" if score>=95 else ("GOOD - Minor Cleaning" if score>=85 else "NEEDS CLEANING")
        self.write()

    def write(self):
        with open(REPORT_FILE,"w",encoding="utf-8") as f:
            f.write("VEDAZ AI - PHASE 1 DATASET AUDIT REPORT\n")
            f.write("="*60+"\n\n")
            for k,v in self.summary.items():
                f.write(f"{k}\n{'-'*len(k)}\n")
                if isinstance(v,dict):
                    for a,b in v.items():
                        f.write(f"{a}: {b}\n")
                else:
                    f.write(f"{v}\n")
                f.write("\n")
        with open(SUMMARY_FILE,"w",encoding="utf-8") as f:
            json.dump(self.summary,f,indent=2,ensure_ascii=False)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("dataset")
    a=p.parse_args()
    l=Loader(a.dataset)
    d=l.load()
    aud=Auditor(d,l)
    aud.run()
    print("Done.")
    print(f"Generated: {REPORT_FILE}")
    print(f"Generated: {SUMMARY_FILE}")

if __name__=="__main__":
    main()
