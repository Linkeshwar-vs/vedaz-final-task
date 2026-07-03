import torch
from transformers import BitsAndBytesConfig

# Quantization Settings
LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_4BIT_COMPUTE_DTYPE = torch.float16

# Device
DEVICE_MAP = "auto"

# Transformers BitsAndBytes configuration
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=LOAD_IN_4BIT,
    bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
    bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    bnb_4bit_compute_dtype=BNB_4BIT_COMPUTE_DTYPE,
)

def print_quantization_config():
    print("=" * 60)
    print("VEDAZ AI - QLORA CONFIGURATION")
    print("=" * 60)
    print(f"4-bit Loading        : {LOAD_IN_4BIT}")
    print(f"Quantization Type    : {BNB_4BIT_QUANT_TYPE}")
    print(f"Double Quantization  : {BNB_4BIT_USE_DOUBLE_QUANT}")
    print(f"Compute DType        : {BNB_4BIT_COMPUTE_DTYPE}")
    print(f"Device Map           : {DEVICE_MAP}")

if __name__ == "__main__":
    print_quantization_config()
