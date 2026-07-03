"""
Verifies that the trained LoRA adapter was saved correctly.

Checks

- Adapter directory exists
- adapter_config.json
- adapter_model.safetensors
- tokenizer files
- Load base model
- Load tokenizer
- Load adapter
- Run one generation

Outputs

reports/adapter_verification.json

Run:
python scripts/training/verify_saved_adapter.py

"""

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_DIR = Path("outputs/qwen_lora/final_adapter")

REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(exist_ok=True)

REPORT = {
    "status": "PASS",
    "checks": {},
}


def check(name, passed, message=""):

    REPORT["checks"][name] = {
        "status": "PASS" if passed else "FAIL",
        "message": message,
    }

    if not passed:
        REPORT["status"] = "FAIL"


# ============================================================
# FILE CHECKS
# ============================================================

required_files = [

    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",

]

for file in required_files:

    exists = (ADAPTER_DIR / file).exists()

    check(file, exists)

if REPORT["status"] == "FAIL":

    with open(
        REPORT_DIR / "adapter_verification.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(REPORT, f, indent=4)

    raise SystemExit("Adapter files missing.")

# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

check("tokenizer_load", True)

# ============================================================
# LOAD BASE MODEL
# ============================================================

print("Loading base model...")

bnb = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.float16,

)

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb,

    device_map="auto",

    torch_dtype=torch.float16,

)

check("base_model_load", True)

# ============================================================
# LOAD ADAPTER
# ============================================================

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(

    model,

    ADAPTER_DIR,

)

model.eval()

check("adapter_load", True)

# ============================================================
# TEST GENERATION
# ============================================================

print("Running inference...")

messages = [

    {
        "role": "system",
        "content": "You are Vedaz's AI Vedic astrologer.",
    },

    {
        "role": "user",
        "content": "Will I get a government job?",
    },

]

text = tokenizer.apply_chat_template(

    messages,

    tokenize=False,

    add_generation_prompt=True,

)

inputs = tokenizer(

    text,

    return_tensors="pt",

).to(model.device)

with torch.no_grad():

    output = model.generate(

        **inputs,

        max_new_tokens=120,

        temperature=0.7,

        do_sample=True,

    )

generated_ids = output[:, inputs["input_ids"].shape[1]:]

response = tokenizer.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)[0].strip()


check("generation", True)

REPORT["sample_response"] = response

# ============================================================
# SAVE REPORT
# ============================================================

with open(

    REPORT_DIR / "adapter_verification.json",

    "w",

    encoding="utf-8",

) as f:

    json.dump(REPORT, f, indent=4, ensure_ascii=False)

print("\n===================================================")

print("Adapter Verification Finished")

print("===================================================")

print(json.dumps(REPORT, indent=4, ensure_ascii=False))