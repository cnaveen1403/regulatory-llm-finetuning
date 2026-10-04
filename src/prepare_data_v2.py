import json
from pathlib import Path

from prepare_data import (
    CATEGORIES,
    remove_duplicates,
    validate_dataset,
    print_distribution,
    write_jsonl,
)

DATA_DIR = Path("data")


CONTRASTIVE_EXAMPLES = {
    "SAFETY_REQUIREMENT": [
        "Personnel must wear chemical-resistant aprons while preparing the pesticide mixture.",
        "Workers must use face protection when handling concentrated pesticide.",
        "Applicators must wear protective gloves while transferring the product.",
        "Personnel must follow the required decontamination procedure after pesticide handling.",
    ],
    "LABELING_REQUIREMENT": [
        "The product label must state the maximum number of permitted applications.",
        "The label must identify the required re-entry precautions.",
        "The approved label must specify the permitted treatment frequency.",
        "The packaging must display the required emergency contact information.",
    ],
    "REGISTRATION_REQUIREMENT": [
        "The company must obtain product authorization before introducing the pesticide commercially.",
        "The registration holder must maintain an active authorization for the product.",
        "The applicant must renew the pesticide authorization according to the regulatory schedule.",
        "The pesticide product requires regulatory approval before commercial distribution.",
    ],
    "STORAGE_REQUIREMENT": [
        "The pesticide must be kept in a designated storage facility when not being used.",
        "The product must be protected from moisture while it is stored.",
        "Pesticides must be kept in a secure warehouse away from animal feed.",
        "The storage facility must maintain the product under the conditions specified by the manufacturer.",
    ],
    "APPLICATION_REQUIREMENT": [
        "Applicators must comply with the maximum number of treatments permitted per season.",
        "The pesticide must be applied using the equipment specified for the authorized use.",
        "Operators must observe the required interval between pesticide treatments.",
        "The product must only be used according to its approved treatment conditions.",
    ],
    "ENVIRONMENTAL_REQUIREMENT": [
        "Pesticide use must minimize exposure of aquatic organisms.",
        "Operators must maintain the required buffer zone around sensitive habitats.",
        "The product must not be used where runoff could affect protected ecosystems.",
        "Pesticide application must minimize impacts on beneficial wildlife.",
    ],
    "REPORTING_REQUIREMENT": [
        "The registrant must provide the authority with the required annual compliance report.",
        "Companies must submit specified incident information within the regulatory deadline.",
        "The manufacturer must communicate required surveillance findings to the regulator.",
        "The company must notify the competent authority when a reportable event occurs.",
    ],
}


def load_jsonl(filename):
    path = DATA_DIR / filename

    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def create_contrastive_examples():
    examples = []

    for category, statements in CONTRASTIVE_EXAMPLES.items():
        for statement in statements:
            examples.append(
                {
                    "instruction": "Classify the following regulatory statement.",
                    "input": statement,
                    "output": category,
                }
            )

    return examples


def main():
    print("Loading Dataset v1...")

    train_v1 = load_jsonl("train.jsonl")
    validation_v1 = load_jsonl("validation.jsonl")
    test_v1 = load_jsonl("test.jsonl")

    print(f"v1 train:      {len(train_v1)}")
    print(f"v1 validation: {len(validation_v1)}")
    print(f"v1 test:       {len(test_v1)}")

    new_examples = create_contrastive_examples()

    print(f"\nNew contrastive examples: {len(new_examples)}")

    # Add new examples only to training.
    train_v2 = train_v1 + new_examples

    # Remove any accidental duplicates.
    train_v2 = remove_duplicates(train_v2)

    # Validation and test remain exactly the same.
    validation_v2 = validation_v1
    test_v2 = test_v1

    print("\nValidating Dataset v2...")

    validate_dataset(
        train_v2,
        validation_v2,
        test_v2,
    )

    # Explicitly verify that the test set did not change.
    if test_v1 != test_v2:
        raise ValueError("ERROR: v1 test set was modified.")

    # Explicitly verify that validation did not change.
    if validation_v1 != validation_v2:
        raise ValueError("ERROR: v1 validation set was modified.")

    write_jsonl("v2_train.jsonl", train_v2)
    write_jsonl("v2_validation.jsonl", validation_v2)
    write_jsonl("v2_test.jsonl", test_v2)

    print("\nDataset v2 created successfully.")

    print(f"\nv2 train:      {len(train_v2)}")
    print(f"v2 validation: {len(validation_v2)}")
    print(f"v2 test:       {len(test_v2)}")

    print_distribution("v2 Train", train_v2)
    print_distribution("v2 Validation", validation_v2)
    print_distribution("v2 Test", test_v2)


if __name__ == "__main__":
    main()
