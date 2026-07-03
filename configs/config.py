"""
=========================================================
VEDAZ AI
Master Configuration (LOCKED)
=========================================================

This is the ONLY file that should be edited when changing
training settings.

All other modules import values from here.

Author:
Vedaz AI
"""

from pathlib import Path
import torch

# ==========================================================
# PROJECT
# ==========================================================

PROJECT_NAME = "Vedaz AI"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# DATASET
# ==========================================================

DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_DATASET = DATA_DIR / "training_dataset.jsonl"
VALIDATION_DATASET = DATA_DIR / "validation_dataset.jsonl"

# ==========================================================
# MODEL
# ==========================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

MAX_SEQUENCE_LENGTH = 1024

SEED = 42

# ==========================================================
# OUTPUTS
# ==========================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

FINAL_MODEL_DIR = OUTPUT_DIR / "final_model"

LOG_DIR = OUTPUT_DIR / "logs"

EVALUATION_DIR = OUTPUT_DIR / "evaluation"

# ==========================================================
# QLoRA
# ==========================================================

LOAD_IN_4BIT = True

BNB_4BIT_QUANT_TYPE = "nf4"

BNB_USE_DOUBLE_QUANT = True

COMPUTE_DTYPE = torch.float16

DEVICE_MAP = "auto"

# ==========================================================
# LoRA
# ==========================================================

LORA_R = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.05

LORA_BIAS = "none"

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# ==========================================================
# TRAINING
# ==========================================================

NUM_EPOCHS = 8

LEARNING_RATE = 2e-4

PER_DEVICE_BATCH_SIZE = 1

PER_DEVICE_EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.05

FP16 = True

SAVE_STRATEGY = "epoch"

EVAL_STRATEGY = "epoch"

LOGGING_STEPS = 1

SAVE_TOTAL_LIMIT = 2

REPORT_TO = "none"

LOAD_BEST_MODEL_AT_END = True

# ==========================================================
# INFERENCE
# ==========================================================

MAX_NEW_TOKENS = 512

TEMPERATURE = 0.7

TOP_P = 0.9

DO_SAMPLE = True

# ==========================================================
# EVALUATION
# ==========================================================

COMPUTE_PERPLEXITY = True

SAVE_EVALUATION_JSON = True

# ==========================================================
# DIRECTORIES
# ==========================================================

DIRECTORIES = [
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    FINAL_MODEL_DIR,
    LOG_DIR,
    EVALUATION_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)