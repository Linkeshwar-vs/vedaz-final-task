"""

Run:
python scripts/training/environment_check.py

"""


import json,platform,importlib
from pathlib import Path

def check(pkg):

    try:
        m=importlib.import_module(pkg)
        return {"installed":True,"version":getattr(m,"__version__","unknown")}

    except Exception as e:
        return {"installed":False,"error":str(e)}

r={"system":{"python":platform.python_version(),"platform":platform.platform()}}

try:
    import torch
    r["torch"]={"version":torch.__version__,"cuda_available":torch.cuda.is_available()}

    if torch.cuda.is_available():
        p=torch.cuda.get_device_properties(0)
        gpu={"name":torch.cuda.get_device_name(0),"vram_gb":round(p.total_memory/1024**3,2),"cuda":torch.version.cuda}

        try:
            a=torch.randn((256,256),device="cuda");b=torch.randn((256,256),device="cuda");_=a@b
            torch.cuda.synchronize()

            gpu["smoke_test"]="PASS"
            gpu["peak_memory_mb"]=round(torch.cuda.max_memory_allocated()/1024**2,2)

            torch.cuda.empty_cache()

        except Exception as e:
            gpu["smoke_test"]=str(e)

        r["gpu"]=gpu

except Exception as e:
    r["torch"]={"error":str(e)}

pkgs=["transformers","datasets","accelerate","trl","peft","bitsandbytes","unsloth","huggingface_hub","evaluate"]

r["packages"]={p:check(p) for p in pkgs}
r["recommendation"]={"model":"Qwen2.5-1.5B-Instruct","method":"QLoRA","precision":"FP16","batch_size":1,"gradient_accumulation":8,"max_sequence_length":1024}

ready=r.get("torch",{}).get("cuda_available",False) and r["packages"]["transformers"]["installed"] and r["packages"]["peft"]["installed"]

r["ready_for_training"]=ready

Path("reports").mkdir(parents=True, exist_ok=True)

Path("reports/environment_report.json").write_text(
    json.dumps(r, indent=2),
    encoding="utf-8"
)
print("READY FOR TRAINING" if ready else "NOT READY")
print("Generated environment_report.json")
