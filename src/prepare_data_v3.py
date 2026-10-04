import json
from pathlib import Path

from prepare_data import (
    remove_duplicates,
    validate_dataset,
    print_distribution,
    write_jsonl,
)

DATA_DIR = Path("data")


# These examples specifically target the confusion patterns
# observed during Dataset v2 evaluation.
CONTRASTIVE_EXAMPLES = {
    "SAFETY_REQUIREMENT": [
        "Workers must use face protection when transferring concentrated pesticide.",
        "Personnel must follow the required decontamination procedure after handling the product.",
        "Applicators must use protective clothing while preparing the pesticide mixture.",
        "Personnel must use approved chemical-resistant boots during pesticide handling.",
    ],
    "STORAGE_REQUIREMENT": [
        "The pesticide must be kept in a locked room when it is not being used.",
        "The product must be stored under the temperature conditions specified by the manufacturer.",
        "Pesticide containers must remain tightly closed during storage.",
        "Unused pesticide must be kept separately from medicines and household products.",
    ],
    "LABELING_REQUIREMENT": [
        "The product label must state the permitted application rate.",
        "The approved label must specify the maximum number of treatments allowed.",
        "The label must identify the authorized treatment interval.",
        "The packaging must display the required handling instructions.",
    ],
    "ENVIRONMENTAL_REQUIREMENT": [
        "Pesticide application must minimize risks to aquatic organisms.",
        "Operators must maintain the required distance from protected habitats.",
        "The product must not be used where pesticide runoff could damage ecosystems.",
        "Pesticide application must minimize risks to non-target insects.",
    ],
    "REPORTING_REQUIREMENT": [
        "The registrant must submit annual production quantities to the competent authority.",
        "The company must report required monitoring results to the regulator.",
        "The manufacturer must notify the authority when a reportable incident occurs.",
        "The registrant must provide required compliance information to the regulatory authority.",
    ],
    "REGISTRATION_REQUIREMENT": [
        "The company must obtain product authorization before commercial distribution.",
        "The registrant must maintain an active authorization for the pesticide.",
        "The applicant must renew the product registration before its expiration.",
        "The pesticide requires regulatory approval before it can be marketed.",
    ],
    "APPLICATION_REQUIREMENT": [
        "The operator must not exceed the permitted application rate.",
        "Applicators must follow the authorized treatment interval when using the product.",
        "The pesticide must be applied using the equipment approved for the intended use.",
        "Operators must comply with the maximum number of treatments permitted per season.",
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
    print("Loading Dataset v2...")

    train_v2 = load_jsonl("v2_train.jsonl")
    validation_v2 = load_jsonl("v2_validation.jsonl")
    test_v2 = load_jsonl("v2_test.jsonl")

    print(f"v2 train:      {len(train_v2)}")
    print(f"v2 validation: {len(validation_v2)}")
    print(f"v2 test:       {len(test_v2)}")

    new_examples = create_contrastive_examples()

    print(f"\nNew contrastive examples: {len(new_examples)}")

    # Add only to training.
    train_v3 = train_v2 + new_examples

    # Remove accidental duplicates.
    train_v3 = remove_duplicates(train_v3)

    # Validation and test remain frozen.
    validation_v3 = validation_v2
    test_v3 = test_v2

    print("\nValidating Dataset v3...")

    validate_dataset(
        train_v3,
        validation_v3,
        test_v3,
    )

    # Explicitly verify validation and test have not changed.
    if validation_v2 != validation_v3:
        raise ValueError("ERROR: validation set was modified.")

    if test_v2 != test_v3:
        raise ValueError("ERROR: test set was modified.")

    write_jsonl("v3_train.jsonl", train_v3)
    write_jsonl("v3_validation.jsonl", validation_v3)
    write_jsonl("v3_test.jsonl", test_v3)

    print("\nDataset v3 created successfully.")

    print(f"\nv3 train:      {len(train_v3)}")
    print(f"v3 validation: {len(validation_v3)}")
    print(f"v3 test:       {len(test_v3)}")

    print_distribution("v3 Train", train_v3)
    print_distribution("v3 Validation", validation_v3)
    print_distribution("v3 Test", test_v3)


if __name__ == "__main__":
    main()
