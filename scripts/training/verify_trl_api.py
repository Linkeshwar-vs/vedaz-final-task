"""
Checks the installed TRL version and inspects every API that
our training pipeline depends on.

This script is the final verification before implementing
trainer_builder.py.

Run:
python scripts/training/verify_trl_api.py

"""

import inspect
import json
import sys
from pathlib import Path

REPORT = {}
LINES = []


def title(name):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    LINES.append("\n" + "=" * 70)
    LINES.append(name)
    LINES.append("=" * 70)


def line(name, value):
    print(f"{name:<35} {value}")
    LINES.append(f"{name:<35} {value}")


# ---------------------------------------------------------
# Python
# ---------------------------------------------------------

REPORT["python"] = sys.version

title("SYSTEM")

line("Python", sys.version.split()[0])

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

title("IMPORTS")

from transformers import (
    AutoTokenizer,
    TrainingArguments,
)

from datasets import load_dataset

from peft import LoraConfig

from trl import (
    SFTTrainer,
    SFTConfig,
)

line("Transformers", "OK")
line("TRL", "OK")
line("PEFT", "OK")
line("Datasets", "OK")

REPORT["imports"] = "PASS"

# ---------------------------------------------------------
# SFTTrainer
# ---------------------------------------------------------

title("SFTTrainer")

sig = inspect.signature(SFTTrainer.__init__)

trainer_args = list(sig.parameters.keys())

REPORT["trainer_parameters"] = trainer_args

for arg in trainer_args:
    line(arg, "✓")

# ---------------------------------------------------------
# SFTConfig
# ---------------------------------------------------------

title("SFTConfig")

cfg = inspect.signature(SFTConfig)

cfg_args = list(cfg.parameters.keys())

REPORT["sftconfig_parameters"] = cfg_args

for arg in cfg_args[:60]:
    line(arg, "✓")

# ---------------------------------------------------------
# TrainingArguments
# ---------------------------------------------------------

title("TrainingArguments")

ta = inspect.signature(TrainingArguments)

ta_args = list(ta.parameters.keys())

REPORT["training_arguments"] = ta_args

important = [
    "output_dir",
    "learning_rate",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "fp16",
    "logging_steps",
    "save_strategy",
    "eval_strategy",
]

for item in important:

    line(
        item,
        "YES" if item in ta_args else "NO"
    )

# ---------------------------------------------------------
# LoRA
# ---------------------------------------------------------

title("LoRA")

lora = inspect.signature(LoraConfig)

REPORT["lora_parameters"] = list(lora.parameters.keys())

for p in [
    "r",
    "lora_alpha",
    "target_modules",
    "lora_dropout",
    "bias",
    "task_type",
]:
    line(
        p,
        "YES" if p in lora.parameters else "NO"
    )

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

title("DATASET")

dataset = load_dataset(
    "json",
    data_files="data/processed/training_dataset.jsonl",
    split="train",
)

line("Rows", len(dataset))

line("Columns", dataset.column_names)

REPORT["dataset_columns"] = dataset.column_names

if "messages" in dataset.column_names:
    line("messages column", "FOUND")
else:
    line("messages column", "MISSING")

# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

title("QWEN TOKENIZER")

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    trust_remote_code=True,
)

line(
    "chat_template",
    tokenizer.chat_template is not None
)

REPORT["chat_template_exists"] = (
    tokenizer.chat_template is not None
)

REPORT["chat_template"] = hasattr(
    tokenizer,
    "apply_chat_template",
)

line(
    "apply_chat_template",
    REPORT["chat_template"]
)

# ---------------------------------------------------------
# Decision
# ---------------------------------------------------------

title("FINAL DECISION")

processing = "processing_class" in trainer_args
formatting = "formatting_func" in trainer_args

line("processing_class", processing)
line("formatting_func", formatting)

if processing:
    recommendation = (
        "Use processing_class=tokenizer"
    )
else:
    recommendation = (
        "Use tokenizer="
    )

line("Recommended Trainer API", recommendation)

REPORT["recommendation"] = recommendation

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

Path("reports").mkdir(exist_ok=True)

with open(
    "reports/sfttrainer_api_report.json",
    "w",
    encoding="utf8",
) as f:
    json.dump(REPORT, f, indent=4)

with open(
    "reports/sfttrainer_api_report.txt",
    "w",
    encoding="utf8",
) as f:
    f.write("\n".join(LINES))

print("\n")
print("=" * 70)
print("REPORT GENERATED")
print("=" * 70)
print("reports/sfttrainer_api_report.json")
print("reports/sfttrainer_api_report.txt")