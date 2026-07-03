# Vedaz AI – Qwen2.5 Fine-Tuning

This repository contains my solution for the Vedaz AI technical assessment.

Objective : To fine-tune **Qwen2.5-1.5B-Instruct** on the provided conversational dataset while preserving the model's safety behaviour and adapting it to the Vedaz assistant style.

I kept inference simple as the objective of the assessment was to demonstrate the fine-tuning pipeline to showcase what the LoRA adapter had learned. In a production deployment, I would combine the fine-tuned model with a strong system prompt, targeted few-shot examples for difficult cases, and inference-time guardrails to achieve more consistent Vedaz-specific behavior.

## Requirements

- Python 3.10+
- NVIDIA GPU (recommended)
- CUDA-compatible PyTorch
- 4 GB+ GPU VRAM (minimum tested: RTX 3050 Laptop GPU)

---

## Outputs

The trained artifacts are available under:

outputs/
└── qwen_lora/
    └── final_adapter/

This directory contains:
- adapter_model.safetensors
- adapter_config.json
- tokenizer.json
- tokenizer_config.json


Training reports and verification results are available in the `reports/` directory, including:

- Dataset audit
- Dataset verification
- Environment checks
- Adapter verification
- Evaluation metrics

## Project Structure

```
VEDAZ-FINAL-TASK/
│
├── configs/                 # Training, LoRA and quantization configurations
├── data/
│   ├── raw/                 # Original dataset ( Provided for the task )
│   └── processed/           # Cleaned dataset and train/validation splits
│
├── outputs/                 # Trained adapters, evaluation results and transcripts
├── reports/                 # Dataset analysis and verification reports
│
├── scripts/
│   ├── preprocessing/       # Dataset auditing and preprocessing scripts
│   └── training/            # Training, evaluation and inference scripts
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Pipeline

The project follows the complete fine-tuning workflow:

### 1. Dataset Analysis

- Audited the provided dataset
- Checked conversation structure
- Detected duplicate conversations
- Generated dataset statistics
- Analysed language and safety coverage

### 2. Dataset Preprocessing

- Removed duplicate conversations
- Normalized message formatting
- Preserved the native `messages` format required by TRL
- Converted the dataset into Hugging Face compatible JSONL files
- Split the dataset into **88% training** and **12% validation**

### 3. Model Training

- Fine-tuned **Qwen2.5-1.5B-Instruct**
- Used **QLoRA** with **PEFT LoRA adapters**
- Used **TRL SFTTrainer**
- Applied **4-bit NF4 quantization**
- Saved only the LoRA adapters

### 4. Evaluation

- Evaluated on the validation set
- Computed validation loss and perplexity
- Verified adapter loading and inference

---

## Main Scripts

| Script | Purpose |
|---------|---------|
| `dataset_audit.py` | Initial dataset analysis |
| `deep_audit.py` | Detailed dataset statistics |
| `prepare_dataset.py` | Cleans dataset and creates train/validation JSONL files |
| `verify_dataset.py` | Verifies processed dataset before training |
| `environment_check.py` | Checks Python, CUDA and package compatibility |
| `api_check.py` | Verifies TRL, Transformers and PEFT APIs |
| `trainer_builder.py` | Builds tokenizer, model, datasets and SFTTrainer |
| `train_qwen.py` | Runs LoRA fine-tuning |
| `verify_saved_adapter.py` | Verifies saved adapter files |
| `evaluate_model.py` | Evaluates the trained model |
| `inference_test.py` | Interactive CLI for chatting with the model |

---

# Running the Project

## 1. Clone the Repository

```bash
git clone https://github.com/Linkeshwar-vs/vedaz-final-task.git
cd vedaz-final-task
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

First, install the **GPU version of PyTorch** appropriate for your CUDA version.

Example (CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For other CUDA versions, refer to the official PyTorch installation guide:

https://pytorch.org/get-started/locally/

Then install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

---

## 4. Dataset Preparation

```bash
python scripts/preprocessing/dataset_audit.py data/raw/dataset.json

python scripts/preprocessing/deep_audit.py data/raw/dataset.json

python scripts/preprocessing/prepare_dataset.py data/raw/dataset.json

python scripts/preprocessing/verify_dataset.py data/processed/cleaned_dataset.json data/processed/training_dataset.jsonl data/processed/validation_dataset.jsonl
```

> **Note:** Replace `^` with `\` on Linux/macOS.

---

## 5. Environment Verification

```bash
python scripts/training/environment_check.py

python scripts/training/api_check.py

python scripts/training/verify_trl_api.py
```

---

## 6. Train the Model

```bash
python scripts/training/train_qwen.py
```

---

## 7. Verify the Saved Adapter

```bash
python scripts/training/verify_saved_adapter.py
```

---

## 8. Evaluate the Model

```bash
python scripts/training/evaluate_model.py
```

---

## 9. Run Interactive Inference

```bash
python scripts/training/inference_test.py
```

## Design Choices

- **Qwen2.5-1.5B-Instruct** was chosen because it provides strong multilingual instruction-following capabilities and native chat template support.

- **Supervised Fine-Tuning (SFT)** was used since the dataset contains demonstration conversations rather than preference pairs.

- **QLoRA + LoRA** enables fine-tuning on consumer GPUs while keeping memory usage low.

- **4-bit NF4 quantization** further reduces GPU memory requirements with minimal impact on model quality.

- **Adapter-only checkpoints** make the final model lightweight and easier to deploy.

---

## Observations

The fine-tuned model is able to produce responses that reflect several of the safety guidelines and conversational patterns present in the training data during qualitative testing. However, the provided dataset contains only **44 conversations** across English, Hindi, and Hinglish, which provides limited coverage of the desired Vedaz persona and conversational behaviours. A 1.5B model has a lot of prior behavior that isn't easy to overwrite with a tiny LoRA.

In order to Produce a production-ready Vedaz assistant indistinguishable from the company's internal model we would require even more dataset


For deployment, the trained LoRA adapter can be served together with the base Qwen model using **vLLM** on a GPU-enabled VPS.


## Author

Linkeshwar VS

BTech Computer Science Engineering (AIML)
VIT Chennai
