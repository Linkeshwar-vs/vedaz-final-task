"""
Main training script.

Pipeline:
1. Build Trainer
2. Train
3. Save LoRA Adapter
4. Save Tokenizer
5. Save Metrics

Run:
python scripts/training/train_qwen.py

"""

import json
import os
import time
from pathlib import Path

import torch

from trainer_builder import (
    build_trainer,
    load_tokenizer,
)

# ------------------------------------------------------------
# OUTPUT PATHS
# ------------------------------------------------------------

OUTPUT_DIR = Path("outputs/qwen_lora")
FINAL_DIR = OUTPUT_DIR / "final_adapter"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# GPU INFO
# ------------------------------------------------------------

def print_gpu_info():

    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE")
        return

    print("=" * 60)
    print("GPU INFORMATION")
    print("=" * 60)

    print(f"GPU : {torch.cuda.get_device_name(0)}")

    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"VRAM : {total:.2f} GB")

    print("=" * 60)


# ------------------------------------------------------------
# SAVE TRAINING METRICS
# ------------------------------------------------------------

def save_metrics(train_result, elapsed):

    metrics = dict(train_result.metrics)

    metrics["training_time_seconds"] = round(elapsed, 2)

    if torch.cuda.is_available():

        metrics["peak_gpu_memory_mb"] = round(
            torch.cuda.max_memory_allocated() / 1024**2,
            2,
        )

    with open(
        REPORT_DIR / "training_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(metrics, f, indent=4)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("VEDAZ AI")
    print("QWEN 2.5 SFT TRAINING")
    print("=" * 70)

    print_gpu_info()

    print("\nBuilding trainer...")

    trainer = build_trainer()

    tokenizer = load_tokenizer()

    print("Done.")

    print("\nStarting Training...\n")

    start = time.time()

    train_result = trainer.train()

    elapsed = time.time() - start

    print("\nTraining Finished.")

    print("\nSaving Adapter...")

    trainer.save_model(str(FINAL_DIR))

    print("Done.")

    print("\nSaving Tokenizer...")

    tokenizer.save_pretrained(str(FINAL_DIR))

    print("Done.")

    print("\nSaving Metrics...")

    save_metrics(train_result, elapsed)

    print("Done.")

    print("\nTraining Summary")
    print("-" * 40)

    print(f"Runtime : {elapsed:.2f} seconds")

    if "train_loss" in train_result.metrics:
        print(f"Train Loss : {train_result.metrics['train_loss']:.4f}")

    if torch.cuda.is_available():

        print(
            "Peak GPU Memory : "
            f"{torch.cuda.max_memory_allocated()/1024**2:.2f} MB"
        )

    print("\nOutput Directory")
    print(FINAL_DIR)

    print("\nTraining Complete.")


# ------------------------------------------------------------

if __name__ == "__main__":

    main()