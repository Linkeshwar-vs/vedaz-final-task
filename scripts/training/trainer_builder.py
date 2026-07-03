"""
Builds all objects required for training:

- Tokenizer
- Quantized Model
- LoRA Config
- SFTConfig
- HuggingFace Dataset
- SFTTrainer

Run:
python scripts/training/trainer_builder.py

"""

from datasets import load_dataset

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    TaskType,
)

from trl import (
    SFTTrainer,
    SFTConfig,
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_FILE = "data/processed/training_dataset.jsonl"
VALID_FILE = "data/processed/validation_dataset.jsonl"

OUTPUT_DIR = "outputs/qwen_lora"

MAX_SEQ_LEN = 1024


# ============================================================
# TOKENIZER
# ============================================================

def load_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


# ============================================================
# QUANTIZATION
# ============================================================

def build_bnb_config():

    return BitsAndBytesConfig(

        load_in_4bit=True,

        bnb_4bit_quant_type="nf4",

        bnb_4bit_use_double_quant=True,

        bnb_4bit_compute_dtype=torch.float16,

    )


# ============================================================
# MODEL
# ============================================================

def load_model():

    model = AutoModelForCausalLM.from_pretrained(

        MODEL_NAME,

        quantization_config=build_bnb_config(),

        device_map="auto",

        torch_dtype=torch.float16,

        trust_remote_code=True,

    )

    model.config.use_cache = False

    return model


# ============================================================
# DATASET
# ============================================================

def load_training_dataset():

    return load_dataset(

        "json",

        data_files=TRAIN_FILE,

        split="train",

    )


def load_validation_dataset():

    return load_dataset(

        "json",

        data_files=VALID_FILE,

        split="train",

    )


# ============================================================
# LoRA
# ============================================================

def build_lora_config():

    return LoraConfig(

        r=16,

        lora_alpha=32,

        lora_dropout=0.05,

        bias="none",

        task_type=TaskType.CAUSAL_LM,

        target_modules=[

            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",

        ],

    )


# ============================================================
# TRAINING CONFIG
# ============================================================

def build_training_config():

    return SFTConfig(

        output_dir=OUTPUT_DIR,

        learning_rate=2e-5,

        per_device_train_batch_size=1,

        per_device_eval_batch_size=1,

        gradient_accumulation_steps=8,

        num_train_epochs=3,

        logging_steps=1,

        eval_strategy="epoch",

        save_strategy="epoch",

        save_total_limit=2,

        fp16=True,

        bf16=False,

        max_length=MAX_SEQ_LEN,

        packing=False,

        assistant_only_loss=False,

        report_to="none",

    )


# ============================================================
# TRAINER
# ============================================================

def build_trainer():

    tokenizer = load_tokenizer()

    model = load_model()

    train_dataset = load_training_dataset()

    validation_dataset = load_validation_dataset()

    trainer = SFTTrainer(

        model=model,

        args=build_training_config(),

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        peft_config=build_lora_config(),

    )

    return trainer


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    trainer = build_trainer()

    print("\nTrainer built successfully.\n")

    print(trainer)