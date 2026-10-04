import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_MODEL_PATH = Path("models/regulatory-qwen-lora-v4")

TEST_FILE = Path("data/test.jsonl")


def load_model():
    print(f"Loading base model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_PATH)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        LORA_MODEL_PATH,
    )

    model.eval()

    return tokenizer, model


def load_test_data():
    examples = []

    with TEST_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            examples.append(json.loads(line))

    return examples


def predict(
    tokenizer,
    model,
    statement: str,
) -> str:

    prompt = f"""
Classify the following regulatory statement into exactly
one of these categories:

SAFETY_REQUIREMENT
LABELING_REQUIREMENT
REGISTRATION_REQUIREMENT
STORAGE_REQUIREMENT
APPLICATION_REQUIREMENT
ENVIRONMENTAL_REQUIREMENT
REPORTING_REQUIREMENT

Statement:
{statement}

Return only the category name.
"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return response


def main():

    tokenizer, model = load_model()

    examples = load_test_data()

    print(f"\nRunning LoRA model on " f"{len(examples)} test examples...\n")

    for index, example in enumerate(
        examples,
        start=1,
    ):

        prediction = predict(
            tokenizer,
            model,
            example["input"],
        )

        print("=" * 70)

        print(f"Example: {index}")

        print(f"Statement: {example['input']}")

        print(f"Expected:  {example['output']}")

        print(f"Predicted: {prediction}")


if __name__ == "__main__":
    main()
