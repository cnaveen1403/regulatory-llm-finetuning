import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRAIN_FILE = Path("data/v4_train.jsonl")
VALIDATION_FILE = Path("data/v4_validation.jsonl")

OUTPUT_DIR = Path("models/regulatory-qwen-lora-v4")


class RegulatoryDataCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):

        max_length = max(len(example["input_ids"]) for example in examples)

        input_ids = []
        labels = []
        attention_masks = []

        for example in examples:

            input_id = example["input_ids"]
            label = example["labels"]

            padding_length = max_length - len(input_id)

            padded_input_ids = input_id + [self.tokenizer.pad_token_id] * padding_length

            padded_labels = label + [-100] * padding_length

            attention_mask = [1] * len(input_id) + [0] * padding_length

            input_ids.append(padded_input_ids)
            labels.append(padded_labels)
            attention_masks.append(attention_mask)

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
        }


def load_examples(file_path):
    examples = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    return examples


def format_example(example):
    prompt = f"""Instruction:
{example["instruction"]}

Input:
{example["input"]}

Category:
"""

    target = example["output"]

    return prompt, target


def tokenize_example(example, tokenizer):

    prompt, target = format_example(example)

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    target_ids = tokenizer(
        target,
        add_special_tokens=False,
    )["input_ids"]

    input_ids = prompt_ids + target_ids

    labels = [-100] * len(prompt_ids) + target_ids

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def prepare_dataset(file_path, tokenizer):

    examples = load_examples(file_path)

    dataset = Dataset.from_list(examples)

    dataset = dataset.map(
        lambda example: tokenize_example(
            example,
            tokenizer,
        )
    )

    return dataset


def main():

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    model.to(device)

    print("Preparing LoRA...")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    print("\nPreparing datasets...")

    train_dataset = prepare_dataset(
        TRAIN_FILE,
        tokenizer,
    )

    validation_dataset = prepare_dataset(
        VALIDATION_FILE,
        tokenizer,
    )

    print(f"Train examples:      {len(train_dataset)}")
    print(f"Validation examples: {len(validation_dataset)}")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )

    data_collator = RegulatoryDataCollator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
    )

    print("\nStarting training...")

    trainer.train()

    print("\nSaving model...")

    trainer.save_model(str(OUTPUT_DIR))

    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"\nModel saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
