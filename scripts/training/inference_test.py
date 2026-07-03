"""
Commands
--------
exit      -> Quit
quit      -> Quit
clear     -> Clear conversation history
history   -> Print conversation history
save      -> Save conversation to transcript


Run:
python scripts/training/inference_test.py

====================================================================
"""

import json

from datetime import datetime
from pathlib import Path

import torch

from peft import PeftModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_PATH = "outputs/qwen_lora/final_adapter"

TRANSCRIPT_DIR = Path("outputs/transcripts")
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    """You are Vedaz's AI Vedic astrologer using the Lahiri ayanamsa. Always reply in the same language and register as the user (English, Hindi, or Hinglish). Be compassionate, balanced, and non-fatalistic. Never predict death, lifespan, serious illness, guaranteed financial success, lottery numbers, or guaranteed relationship outcomes. Never use fear or certainty. If birth details are required, ask for date of birth, birth time, and birth place before providing an astrological interpretation. Present astrology as guidance about tendencies and timing, not certainties. Offer remedies only as optional spiritual practices, never as guaranteed solutions. For health, legal, financial, or mental health concerns, encourage appropriate qualified professionals. If the user expresses suicidal thoughts or severe emotional distress, prioritize their safety and avoid astrological analysis."""
)


MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Done.")

# ============================================================
# LOAD MODEL
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

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
)

model.eval()

print("Done.\n")

# ============================================================
# CHAT HISTORY
# ============================================================

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
]

# ============================================================
# SAVE TRANSCRIPT
# ============================================================

def save_transcript():

    filename = datetime.now().strftime(
        "vedaz_chat_%Y%m%d_%H%M%S.json"
    )

    path = TRANSCRIPT_DIR / filename

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            messages,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nTranscript saved to:\n{path}\n")

# ============================================================
# HEADER
# ============================================================

# ============================================================
# HELP MENU
# ============================================================

def print_help():

    print("\n" + "=" * 70)
    print("VEDAZ AI")
    print("Fine-tuned Qwen2.5 + LoRA")
    print("=" * 70)

    print("\nAvailable Commands\n")

    print(f"{'help':<12} - Show this help menu")
    print(f"{'history':<12} - Display current conversation history")
    print(f"{'clear':<12} - Clear the current conversation")
    print(f"{'save':<12} - Save the conversation as a JSON transcript")
    print(f"{'exit':<12} - Exit Vedaz AI")
    print(f"{'quit':<12} - Exit Vedaz AI")

    print("\nType your message below to start chatting.")
    print("=" * 70 + "\n")


# Display at startup
print_help()

print("=" * 70)
print("VEDAZ AI")
print("Fine-tuned Qwen2.5 + LoRA")
print("=" * 70)

print("Commands:\n\n")
print(f"{'help':<12} - Show this help menu")
print(f"{'history':<12} - Display current conversation history")
print(f"{'clear':<12} - Clear the current conversation")
print(f"{'save':<12} - Save the conversation as a JSON transcript")
print(f"{'exit':<12} - Exit Vedaz AI")
print(f"{'quit':<12} - Exit Vedaz AI")

# ============================================================
# CHAT LOOP
# ============================================================

try:

    while True:

        user_input = input("\nYou : ").strip()

        if user_input.lower() == "help":
            print_help()
            continue

        if user_input.lower() in ("exit", "quit"):
            print("\nThank you for using Vedaz AI.")
            print("\nGoodbye.\n")
            break

        if user_input.lower() == "clear":

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            ]

            print("Conversation history cleared.")
            continue

        if user_input.lower() == "history":

            print()

            for msg in messages:

                print(
                    f"[{msg['role'].upper()}]\n{msg['content']}\n"
                )

            continue

        if user_input.lower() == "save":

            save_transcript()
            continue

        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():

            outputs = model.generate(

                **inputs,

                max_new_tokens=MAX_NEW_TOKENS,

                temperature=TEMPERATURE,

                top_p=TOP_P,

                do_sample=DO_SAMPLE,

                pad_token_id=tokenizer.pad_token_id,

                eos_token_id=tokenizer.eos_token_id,

            )

        # ----------------------------------------------------
        # Decode ONLY the generated tokens
        # ----------------------------------------------------

        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]

        response = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        print("\nVedaz :", response)

        messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

except KeyboardInterrupt:

    print("\n\nInterrupted.")

    answer = input("Save conversation? (y/n): ")

    if answer.lower().startswith("y"):
        save_transcript()

    print("\nGoodbye.")