import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer

TRAIN_FILE = Path("data/train.jsonl")
VALIDATION_FILE = Path("data/validation.jsonl")
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


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


def collate_batch(examples, tokenizer):

    max_length = max(len(example["input_ids"]) for example in examples)

    input_ids = []
    labels = []
    attention_masks = []

    for example in examples:

        input_id = example["input_ids"]
        label = example["labels"]

        padding_length = max_length - len(input_id)

        padded_input_ids = input_id + [tokenizer.pad_token_id] * padding_length

        padded_labels = label + [-100] * padding_length

        attention_mask = [1] * len(input_id) + [0] * padding_length

        input_ids.append(padded_input_ids)
        labels.append(padded_labels)
        attention_masks.append(attention_mask)

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_masks),
    }


def main():

    train_examples = load_examples(TRAIN_FILE)
    validation_examples = load_examples(VALIDATION_FILE)

    print(f"Training examples:   {len(train_examples)}")
    print(f"Validation examples: {len(validation_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = Dataset.from_list(train_examples)
    validation_dataset = Dataset.from_list(validation_examples)

    train_dataset = train_dataset.map(
        lambda example: tokenize_example(example, tokenizer)
    )

    validation_dataset = validation_dataset.map(
        lambda example: tokenize_example(example, tokenizer)
    )

    batch_examples = [train_dataset[i] for i in range(4)]

    batch = collate_batch(batch_examples, tokenizer)

    print("\nBATCH")
    print("=" * 60)

    print("input_ids shape:")
    print(batch["input_ids"].shape)

    print("\nlabels shape:")
    print(batch["labels"].shape)

    print("\nattention_mask shape:")
    print(batch["attention_mask"].shape)

    print("\ninput_ids:")
    print(batch["input_ids"])

    print("\nlabels:")
    print(batch["labels"])

    print("\nattention_mask:")
    print(batch["attention_mask"])


if __name__ == "__main__":
    main()
