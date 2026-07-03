"""
API Compatibility Check
=========================================================

Purpose
-------
Checks whether the installed versions of
Transformers, TRL, PEFT, Torch and BitsAndBytes
support the APIs required for training.

Run : python scripts/training/api_check.py

"""

import json
import inspect
import platform
import importlib
from pathlib import Path

report = {}

# =====================================================
# Helper
# =====================================================

def get_version(package):
    try:
        module = importlib.import_module(package)
        return getattr(module, "__version__", "Unknown")
    except Exception as e:
        return f"NOT INSTALLED ({e})"


# =====================================================
# System
# =====================================================

print("=" * 60)
print("VEDAZ AI - PHASE 3 API CHECK")
print("=" * 60)

report["system"] = {
    "python": platform.python_version(),
    "platform": platform.platform(),
}

# =====================================================
# Package Versions
# =====================================================

packages = [
    "torch",
    "transformers",
    "trl",
    "peft",
    "datasets",
    "bitsandbytes",
    "accelerate",
]

report["versions"] = {}

print("\nInstalled Packages")
print("-" * 60)

for pkg in packages:

    version = get_version(pkg)

    report["versions"][pkg] = version

    print(f"{pkg:15} : {version}")

# =====================================================
# Torch
# =====================================================

print("\nTorch")
print("-" * 60)

try:

    import torch

    torch_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
    }

    if torch.cuda.is_available():

        torch_info["gpu_name"] = torch.cuda.get_device_name(0)

    report["torch"] = torch_info

    for k, v in torch_info.items():
        print(f"{k:20}: {v}")

except Exception as e:

    report["torch_error"] = str(e)

# =====================================================
# Transformers
# =====================================================

print("\nTransformers")
print("-" * 60)

try:

    from transformers import (
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )

    report["transformers"] = {
        "AutoTokenizer": True,
        "BitsAndBytesConfig": True,
        "TrainingArguments": True,
    }

    print("✓ AutoTokenizer")
    print("✓ BitsAndBytesConfig")
    print("✓ TrainingArguments")

except Exception as e:

    report["transformers_error"] = str(e)

# =====================================================
# TRL
# =====================================================

print("\nTRL")
print("-" * 60)

try:

    from trl import SFTTrainer

    sig = inspect.signature(SFTTrainer.__init__)

    params = list(sig.parameters.keys())

    report["trl"] = {
        "constructor_parameters": params
    }

    print("Constructor Parameters:\n")

    for p in params:
        print("  ", p)

except Exception as e:

    report["trl_error"] = str(e)

# =====================================================
# PEFT
# =====================================================

print("\nPEFT")
print("-" * 60)

try:

    from peft import LoraConfig

    sig = inspect.signature(LoraConfig.__init__)

    params = list(sig.parameters.keys())

    report["peft"] = {
        "LoraConfig_parameters": params
    }

    print("LoraConfig Parameters:\n")

    for p in params:
        print(" ", p)

except Exception as e:

    report["peft_error"] = str(e)

# =====================================================
# Chat Template
# =====================================================

print("\nQwen Tokenizer Check")
print("-" * 60)

try:

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True,
    )

    has_chat_template = hasattr(tokenizer, "apply_chat_template")

    report["chat_template"] = has_chat_template

    print("apply_chat_template :", has_chat_template)

except Exception as e:

    report["chat_template_error"] = str(e)

# =====================================================
# Save Report
# =====================================================

Path("reports").mkdir(parents=True, exist_ok=True)

with open("reports/api_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

print("\n" + "=" * 60)
print("API report written to api_report.json")
print("=" * 60)