from transformers import TrainingArguments
from config import OUTPUT_DIR

TRAINING_ARGS = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=8,
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    weight_decay=0.01,
    warmup_ratio=0.05,
    fp16=True,
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to="none",
    seed=42,
)
