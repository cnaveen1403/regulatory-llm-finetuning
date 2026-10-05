import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRAIN_FILE = Path("data/v4_train.jsonl")
BOUNDARY_TRAIN_FILE = Path("data/boundary_train.jsonl")
VALIDATION_FILE = Path("data/v4_validation.jsonl")

OUTPUT_DIR = Path("models/regulatory-qwen-lora-classifier-v1")


LABELS = [
    "SAFETY_REQUIREMENT",
    "LABELING_REQUIREMENT",
    "REGISTRATION_REQUIREMENT",
    "STORAGE_REQUIREMENT",
    "APPLICATION_REQUIREMENT",
    "ENVIRONMENTAL_REQUIREMENT",
    "REPORTING_REQUIREMENT",
]

LABEL2ID = {label: index for index, label in enumerate(LABELS)}

ID2LABEL = {index: label for index, label in enumerate(LABELS)}


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def main():

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------
    # Classification model
    # ---------------------------------------------------------

    print("Loading classification model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        torch_dtype=torch.float32,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    # ---------------------------------------------------------
    # LoRA configuration
    # ---------------------------------------------------------

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["score"],
    )

    print("\nApplying LoRA...")

    model = get_peft_model(
        model,
        lora_config,
    )

    print("\nTrainable parameters:")

    model.print_trainable_parameters()

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    print("\nLoading datasets...")

    train_records = load_jsonl(TRAIN_FILE)
    boundary_records = load_jsonl(BOUNDARY_TRAIN_FILE)
    validation_records = load_jsonl(VALIDATION_FILE)

    train_records.extend(boundary_records)

    print(f"Original train examples:  {len(train_records) - len(boundary_records)}")
    print(f"Boundary train examples:  {len(boundary_records)}")
    print(f"Total train examples:     {len(train_records)}")
    print(f"Validation examples:      {len(validation_records)}")

    for record in train_records:
        record["label"] = LABEL2ID[record["output"]]

    for record in validation_records:
        record["label"] = LABEL2ID[record["output"]]

    train_dataset = Dataset.from_list(train_records)
    validation_dataset = Dataset.from_list(validation_records)

    # ---------------------------------------------------------
    # Tokenization
    # ---------------------------------------------------------

    def tokenize(batch):
        return tokenizer(
            batch["input"],
            truncation=True,
            max_length=256,
        )

    print("Tokenizing datasets...")

    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
    )

    validation_dataset = validation_dataset.map(
        tokenize,
        batched=True,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    # ---------------------------------------------------------
    # Training arguments
    # ---------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to="none",
        fp16=False,
        dataloader_pin_memory=False,
        use_cpu=(device == "cpu"),
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    lora_parameters = []
    classifier_parameters = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if "score" in name:
            classifier_parameters.append(parameter)
            print(f"Classifier parameter: {name}")
        else:
            lora_parameters.append(parameter)

    print(f"\nLoRA parameter tensors: {len(lora_parameters)}")
    print(f"Classifier parameter tensors: {len(classifier_parameters)}")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_parameters,
                "lr": 2e-5,
            },
            {
                "params": classifier_parameters,
                "lr": 1e-3,
            },
        ],
        weight_decay=0.01,
    )

    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    print("\nStarting LoRA classification training...\n")

    trainer.train()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    print("\nSaving LoRA classification model...")

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"\nModel saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
