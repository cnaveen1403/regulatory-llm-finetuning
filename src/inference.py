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
Classify the following pesticide regulatory statement into exactly ONE category.

Use these definitions carefully:

SAFETY_REQUIREMENT:
Requirements that protect people, workers, operators, or handlers from pesticide exposure.

STORAGE_REQUIREMENT:
Requirements governing how the pesticide or its container must be kept, stored, secured, or separated while not in use.

APPLICATION_REQUIREMENT:
Requirements governing what the operator or applicator must do when applying or using the pesticide.

LABELING_REQUIREMENT:
Requirements describing information, instructions, warnings, rates, or restrictions that must appear on the product label or packaging.

REPORTING_REQUIREMENT:
Requirements to report, submit, notify, or provide information to a regulatory authority.

REGISTRATION_REQUIREMENT:
Requirements to obtain, maintain, renew, or have regulatory authorization or registration before marketing or distributing the product.

ENVIRONMENTAL_REQUIREMENT:
Requirements intended to protect the environment, ecosystems, wildlife, aquatic organisms, or non-target organisms.

Return ONLY the category name.

Allowed categories:
SAFETY_REQUIREMENT
STORAGE_REQUIREMENT
APPLICATION_REQUIREMENT
LABELING_REQUIREMENT
REPORTING_REQUIREMENT
REGISTRATION_REQUIREMENT
ENVIRONMENTAL_REQUIREMENT

Statement:
{statement}
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
