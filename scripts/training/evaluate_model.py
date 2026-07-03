"""
Evaluates the trained LoRA adapter on the validation dataset.

Outputs:

outputs/qwen_lora/

    evaluation_metrics.json
    evaluation_report.txt

Run:
python scripts/training/evaluate_model.py

====================================================================
"""

import json
import math
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel

from trl import SFTTrainer, SFTConfig


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

VALID_DATASET = "data/processed/validation_dataset.jsonl"

ADAPTER_DIR = "outputs/qwen_lora/final_adapter"

OUTPUT_DIR = Path("outputs/qwen_lora")

OUTPUT_DIR.mkdir(exist_ok=True)

REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_DIR,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Done.")


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\nLoading base model...")

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.float16,

)

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto",

    torch_dtype=torch.float16,

)

print("Done.")


# ============================================================
# LOAD ADAPTER
# ============================================================

print("\nLoading LoRA adapter...")

model = PeftModel.from_pretrained(

    model,

    ADAPTER_DIR,

)

model.eval()

print("Done.")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading validation dataset...")

train_dataset = load_dataset(
    "json",
    data_files="data/processed/validation_dataset.jsonl",
    split="train",
)

validation_dataset = load_dataset(
    "json",
    data_files="data/processed/validation_dataset.jsonl",
    split="train",
)

print(f"Validation Samples : {len(validation_dataset)}")


# ============================================================
# EVALUATION CONFIG
# ============================================================

eval_config = SFTConfig(
    output_dir="outputs/qwen_lora/evaluation",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    report_to="none",
    fp16=True,
    packing=False,
)

# ============================================================
# BUILD TRAINER
# ============================================================

print("\nBuilding evaluation trainer...")

trainer = SFTTrainer(
    model=model,
    args=eval_config,
    train_dataset=train_dataset,      # Required by TRL 0.23
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
)

print("Done.")


# ============================================================
# RUN EVALUATION
# ============================================================

print("\nRunning Evaluation...\n")

metrics = trainer.evaluate()

loss = metrics.get("eval_loss", None)

if loss is not None:

    try:

        perplexity = math.exp(loss)

    except OverflowError:

        perplexity = float("inf")

else:

    perplexity = None

metrics["perplexity"] = perplexity

metrics["timestamp"] = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)


# ============================================================
# SAVE JSON
# ============================================================

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

json_file = REPORT_DIR / "evaluation_metrics.json"

with open(json_file, "w", encoding="utf-8") as f:

    json.dump(
        metrics,
        f,
        indent=4,
    )


# ============================================================
# SAVE REPORT
# ============================================================

report_file = REPORT_DIR / "evaluation_report.txt"

with open(report_file, "w", encoding="utf-8") as f:

    f.write("=" * 65 + "\n")
    f.write("VEDAZ AI - MODEL EVALUATION REPORT\n")
    f.write("=" * 65 + "\n\n")

    for key, value in metrics.items():

        f.write(f"{key:<30}: {value}\n")


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("=" * 65)

print("MODEL EVALUATION COMPLETE")

print("=" * 65)

print(f"Validation Loss     : {loss:.4f}")

print(f"Perplexity          : {perplexity:.4f}")

print(f"Runtime             : {metrics.get('eval_runtime',0):.2f} sec")

print(f"Samples/sec         : {metrics.get('eval_samples_per_second',0):.2f}")

print(f"Steps/sec           : {metrics.get('eval_steps_per_second',0):.2f}")

print("\nReports Saved")

print(json_file)

print(report_file)

print("=" * 65)