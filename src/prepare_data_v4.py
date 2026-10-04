import json
from pathlib import Path
from collections import Counter

BASE_TRAIN = Path("data/v3_train.jsonl")
BASE_VALIDATION = Path("data/v3_validation.jsonl")
BASE_TEST = Path("data/test.jsonl")

V4_TRAIN = Path("data/v4_train.jsonl")
V4_VALIDATION = Path("data/v4_validation.jsonl")
V4_TEST = Path("data/v4_test.jsonl")


VALID_CATEGORIES = {
    "SAFETY_REQUIREMENT",
    "LABELING_REQUIREMENT",
    "REGISTRATION_REQUIREMENT",
    "STORAGE_REQUIREMENT",
    "APPLICATION_REQUIREMENT",
    "ENVIRONMENTAL_REQUIREMENT",
    "REPORTING_REQUIREMENT",
}


# These are deliberately minimal/boundary pairs.
# The goal is to teach the model the semantic distinction
# between closely related regulatory requirements.
MINIMAL_PAIRS = [
    # ============================================================
    # SAFETY vs STORAGE
    # ============================================================
    (
        "SAFETY_REQUIREMENT",
        "Personnel must wear a face shield when there is a risk of pesticide splash during handling.",
    ),
    (
        "STORAGE_REQUIREMENT",
        "Pesticide containers must be kept securely closed while in storage.",
    ),
    (
        "SAFETY_REQUIREMENT",
        "Operators must wear protective eyewear while preparing the pesticide mixture.",
    ),
    (
        "STORAGE_REQUIREMENT",
        "Pesticide containers must be kept in a secure storage area when not in use.",
    ),
    (
        "SAFETY_REQUIREMENT",
        "Personnel must use protective clothing to prevent exposure during pesticide handling.",
    ),
    (
        "STORAGE_REQUIREMENT",
        "The pesticide must be kept in a locked storage room when not being used.",
    ),
    (
        "SAFETY_REQUIREMENT",
        "Workers must follow the required decontamination procedure after handling the product.",
    ),
    (
        "STORAGE_REQUIREMENT",
        "The product must remain sealed during storage to prevent leakage.",
    ),
    # ============================================================
    # LABELING vs STORAGE
    # ============================================================
    (
        "LABELING_REQUIREMENT",
        "The product label must specify the required storage conditions.",
    ),
    (
        "STORAGE_REQUIREMENT",
        "The product must be stored under the specified temperature conditions.",
    ),
    (
        "LABELING_REQUIREMENT",
        "The label must state the precautions required for pesticide storage.",
    ),
    (
        "STORAGE_REQUIREMENT",
        "The pesticide must be stored separately from food and animal feed.",
    ),
    (
        "LABELING_REQUIREMENT",
        "The approved label must identify the required storage location.",
    ),
    (
        "STORAGE_REQUIREMENT",
        "Pesticides must be kept in the designated storage location.",
    ),
    (
        "LABELING_REQUIREMENT",
        "The packaging must display the required storage instructions.",
    ),
    (
        "STORAGE_REQUIREMENT",
        "Pesticide containers must remain tightly closed during storage.",
    ),
    # ============================================================
    # LABELING vs APPLICATION
    # ============================================================
    (
        "LABELING_REQUIREMENT",
        "The label must state the maximum permitted application rate.",
    ),
    (
        "APPLICATION_REQUIREMENT",
        "Operators must not exceed the maximum permitted application rate.",
    ),
    (
        "LABELING_REQUIREMENT",
        "The label must specify the authorized treatment interval.",
    ),
    (
        "APPLICATION_REQUIREMENT",
        "Applicators must follow the authorized treatment interval when using the product.",
    ),
    (
        "LABELING_REQUIREMENT",
        "The product label must state the maximum number of treatments permitted per season.",
    ),
    (
        "APPLICATION_REQUIREMENT",
        "Operators must comply with the maximum number of treatments permitted per season.",
    ),
    (
        "LABELING_REQUIREMENT",
        "The label must identify the equipment approved for the intended application.",
    ),
    (
        "APPLICATION_REQUIREMENT",
        "The pesticide must be applied using the equipment approved for the intended use.",
    ),
    # ============================================================
    # REPORTING vs REGISTRATION
    # ============================================================
    (
        "REPORTING_REQUIREMENT",
        "The registrant must submit annual production quantities to the regulatory authority.",
    ),
    (
        "REGISTRATION_REQUIREMENT",
        "The company must obtain product authorization before commercial distribution.",
    ),
    (
        "REPORTING_REQUIREMENT",
        "The company must report required monitoring results to the regulator.",
    ),
    (
        "REGISTRATION_REQUIREMENT",
        "The registrant must maintain an active authorization for the pesticide.",
    ),
    (
        "REPORTING_REQUIREMENT",
        "The manufacturer must notify the authority when a reportable incident occurs.",
    ),
    (
        "REGISTRATION_REQUIREMENT",
        "The applicant must renew the product registration before its expiration.",
    ),
    (
        "REPORTING_REQUIREMENT",
        "The registrant must provide required compliance information to the regulatory authority.",
    ),
    (
        "REGISTRATION_REQUIREMENT",
        "The pesticide requires regulatory approval before it can be marketed.",
    ),
    # ============================================================
    # REPORTING vs STORAGE
    # ============================================================
    (
        "REPORTING_REQUIREMENT",
        "The company must submit annual storage records to the competent authority.",
    ),
    (
        "STORAGE_REQUIREMENT",
        "The company must keep pesticide containers in the designated storage area.",
    ),
    (
        "REPORTING_REQUIREMENT",
        "The registrant must provide the authority with records of pesticide quantities stored.",
    ),
    (
        "STORAGE_REQUIREMENT",
        "Pesticides must be stored separately from incompatible materials.",
    ),
    (
        "REPORTING_REQUIREMENT",
        "The operator must submit storage incident information to the regulatory authority.",
    ),
    (
        "STORAGE_REQUIREMENT",
        "The operator must keep the pesticide secured while it is stored.",
    ),
    # ============================================================
    # APPLICATION vs ENVIRONMENTAL
    # ============================================================
    (
        "APPLICATION_REQUIREMENT",
        "Operators must apply the pesticide at the permitted dosage.",
    ),
    (
        "ENVIRONMENTAL_REQUIREMENT",
        "Operators must apply the pesticide in a manner that minimizes harm to aquatic organisms.",
    ),
    (
        "APPLICATION_REQUIREMENT",
        "Applicators must follow the authorized application interval.",
    ),
    (
        "ENVIRONMENTAL_REQUIREMENT",
        "Applicators must use the product in a manner that minimizes risks to non-target wildlife.",
    ),
    (
        "APPLICATION_REQUIREMENT",
        "The operator must use the pesticide only on the authorized crop.",
    ),
    (
        "ENVIRONMENTAL_REQUIREMENT",
        "The operator must prevent pesticide runoff from reaching protected ecosystems.",
    ),
    (
        "APPLICATION_REQUIREMENT",
        "Operators must follow the permitted application method.",
    ),
    (
        "ENVIRONMENTAL_REQUIREMENT",
        "Operators must maintain the required distance from sensitive environmental areas.",
    ),
]


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_record(category, statement):
    return {
        "instruction": (
            "Classify the following pesticide regulatory statement "
            "into exactly one of the allowed categories."
        ),
        "input": statement,
        "output": category,
    }


def validate_dataset(records, name):
    errors = []

    for i, record in enumerate(records, start=1):
        if not all(key in record for key in ("instruction", "input", "output")):
            errors.append(f"{name}: missing field at record {i}")

        if record["output"] not in VALID_CATEGORIES:
            errors.append(f"{name}: invalid category at record {i}: {record['output']}")

    statements = [record["input"].strip().lower() for record in records]

    duplicates = [
        statement for statement, count in Counter(statements).items() if count > 1
    ]

    if duplicates:
        errors.append(f"{name}: duplicate statements found: {duplicates}")

    if errors:
        raise ValueError("\n".join(errors))


def check_no_leakage(train, validation, test):
    train_text = {r["input"].strip().lower() for r in train}
    validation_text = {r["input"].strip().lower() for r in validation}
    test_text = {r["input"].strip().lower() for r in test}

    train_validation = train_text & validation_text
    train_test = train_text & test_text
    validation_test = validation_text & test_text

    if train_validation:
        raise ValueError(f"Train/validation leakage detected: {train_validation}")

    if train_test:
        raise ValueError(f"Train/test leakage detected: {train_test}")

    if validation_test:
        raise ValueError(f"Validation/test leakage detected: {validation_test}")


def print_distribution(name, records):
    counts = Counter(record["output"] for record in records)

    print(f"\n{name}: {len(records)} examples")

    for category in sorted(VALID_CATEGORIES):
        print(f"  {category}: {counts[category]}")


def main():
    base_train = load_jsonl(BASE_TRAIN)
    validation = load_jsonl(BASE_VALIDATION)
    test = load_jsonl(BASE_TEST)

    print(f"Base v3 train: {len(base_train)}")
    print(f"Validation:     {len(validation)}")
    print(f"Test:            {len(test)}")

    new_examples = [
        make_record(category, statement) for category, statement in MINIMAL_PAIRS
    ]

    existing = {record["input"].strip().lower() for record in base_train}

    unique_new_examples = [
        record
        for record in new_examples
        if record["input"].strip().lower() not in existing
    ]

    train = base_train + unique_new_examples

    validate_dataset(train, "v4_train")
    validate_dataset(validation, "v4_validation")
    validate_dataset(test, "v4_test")

    check_no_leakage(train, validation, test)

    save_jsonl(V4_TRAIN, train)
    save_jsonl(V4_VALIDATION, validation)
    save_jsonl(V4_TEST, test)

    print("\n========================================")
    print("V4 DATASET CREATED")
    print("========================================")

    print_distribution("V4 TRAIN", train)
    print_distribution("V4 VALIDATION", validation)
    print_distribution("V4 TEST", test)

    print(f"\nAdded examples: {len(unique_new_examples)}")
    print(f"V4 train size:  {len(train)}")

    print("\nValidation passed.")
    print("No duplicate statements.")
    print("No train/validation/test leakage.")


if __name__ == "__main__":
    main()
