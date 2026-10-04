import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("data")
SEED = 42

CATEGORIES = {
    "SAFETY_REQUIREMENT",
    "LABELING_REQUIREMENT",
    "REGISTRATION_REQUIREMENT",
    "STORAGE_REQUIREMENT",
    "APPLICATION_REQUIREMENT",
    "ENVIRONMENTAL_REQUIREMENT",
    "REPORTING_REQUIREMENT",
}


TEMPLATES = {
    "SAFETY_REQUIREMENT": [
        "Workers must wear chemical-resistant gloves when handling the pesticide.",
        "Operators must use protective clothing during pesticide application.",
        "Personnel must use approved respiratory protection when applying the product.",
        "Applicators must wear protective eyewear when handling the pesticide.",
        "Workers must use chemical-resistant footwear during treatment.",
        "Personnel handling the product must follow the prescribed safety precautions.",
        "Operators must follow all required personal protection measures.",
        "Anyone applying the product must use the specified protective equipment.",
        "Workers must wash exposed skin after handling the pesticide.",
        "Applicators must avoid direct contact with the pesticide during use.",
        "Personnel must follow the safety instructions provided with the product.",
        "Operators must use the required protective equipment during mixing.",
        "Workers must follow the specified precautions when preparing the pesticide.",
        "Applicators must wear appropriate protective equipment during treatment.",
        "Personnel must follow the required hygiene procedures after handling the product.",
        "Workers must comply with the specified personal safety requirements.",
    ],
    "LABELING_REQUIREMENT": [
        "The product label must include the registration number.",
        "The approved label must display the required precautionary statements.",
        "The packaging must contain the required hazard information.",
        "The product information must clearly state the approved crops.",
        "The label must provide directions for use.",
        "The container label must identify the product manufacturer.",
        "The approved product label must specify application restrictions.",
        "The labeling must contain the required first-aid instructions.",
        "The label must state the authorized uses of the pesticide.",
        "The product label must include the required storage instructions.",
        "The label must identify the active ingredient.",
        "The packaging must display the required warning statements.",
        "The label must specify the approved application rate.",
        "The product label must identify the registration holder.",
        "The label must include the required environmental precautions.",
        "The packaging must contain the legally required product information.",
    ],
    "REGISTRATION_REQUIREMENT": [
        "The product must be registered with the competent authority before marketing.",
        "The registrant must obtain regulatory approval before selling the pesticide.",
        "The pesticide registration must be renewed before it expires.",
        "An application for registration must contain the required supporting documentation.",
        "The product may not be marketed without valid regulatory registration.",
        "The registrant must maintain a valid authorization for the product.",
        "Registration of the pesticide must be completed before commercial distribution.",
        "The company must obtain the required product authorization.",
        "The pesticide must receive regulatory approval before commercial sale.",
        "The registration holder must maintain the product authorization.",
        "A valid registration must be obtained before the product is placed on the market.",
        "The company must submit a registration application to the relevant authority.",
        "The product registration must remain valid throughout the authorized period.",
        "Registration must be renewed according to the applicable regulatory schedule.",
        "The applicant must provide the required documents for product registration.",
        "The pesticide cannot be commercially distributed without registration.",
    ],
    "STORAGE_REQUIREMENT": [
        "The pesticide must be stored in a secure location.",
        "The product must be kept away from food and animal feed.",
        "Pesticides must remain in their original containers during storage.",
        "The product must be stored in an area inaccessible to unauthorized persons.",
        "Storage facilities must protect the pesticide from unauthorized access.",
        "The pesticide must be kept under the storage conditions specified by the label.",
        "Products must be stored separately from incompatible materials.",
        "The pesticide must remain securely stored when not in use.",
        "The product must be stored in a dry and secure area.",
        "Pesticides must be kept away from children and unauthorized personnel.",
        "The storage area must prevent unauthorized access to pesticide products.",
        "The product must remain in a properly sealed container during storage.",
        "Pesticides must be stored according to the conditions specified on the label.",
        "The storage facility must protect the product from contamination.",
        "The pesticide must be stored separately from consumable products.",
        "Unused pesticide must be kept in an approved storage area.",
    ],
    "APPLICATION_REQUIREMENT": [
        "The pesticide must be applied only at the approved application rate.",
        "Applicators must follow the maximum dosage specified on the label.",
        "The product may only be used for the approved application purpose.",
        "The pesticide must not be applied more frequently than permitted.",
        "Application must follow the directions specified on the approved label.",
        "The operator must comply with the permitted application interval.",
        "The product must be applied using the approved application method.",
        "Applicators must follow the authorized use restrictions.",
        "The pesticide must only be applied to approved crops.",
        "Operators must follow the prescribed application rate.",
        "The product must not be applied outside the approved use conditions.",
        "Applicators must comply with the specified treatment frequency.",
        "The pesticide must be used according to the approved directions.",
        "The operator must follow the permitted application timing.",
        "The product must be applied only in the authorized manner.",
        "Applicators must comply with the approved use instructions.",
    ],
    "ENVIRONMENTAL_REQUIREMENT": [
        "The pesticide must not contaminate protected surface water.",
        "Application must comply with restrictions protecting environmentally sensitive areas.",
        "The product must be used in a manner that protects pollinators.",
        "Pesticide application must minimize risks to non-target organisms.",
        "The product must not be applied in areas subject to environmental restrictions.",
        "Operators must comply with measures designed to protect aquatic environments.",
        "Use of the pesticide must comply with environmental protection requirements.",
        "Application must prevent unacceptable environmental contamination.",
        "The pesticide must be used in a manner that protects wildlife.",
        "Application must comply with restrictions intended to protect groundwater.",
        "Operators must take measures to minimize environmental exposure.",
        "The product must not be applied near protected ecological areas.",
        "Pesticide use must minimize risks to beneficial insects.",
        "Application must comply with requirements for protecting aquatic life.",
        "The pesticide must not be used in environmentally restricted zones.",
        "Operators must follow measures designed to prevent environmental contamination.",
    ],
    "REPORTING_REQUIREMENT": [
        "The manufacturer must report specified adverse incidents to the authority.",
        "Registrants must submit the required production information annually.",
        "The company must notify the regulator of reportable incidents.",
        "Required monitoring results must be submitted to the competent authority.",
        "The registrant must report significant changes affecting the product.",
        "Companies must submit the required regulatory reports within the specified period.",
        "Adverse effects must be reported to the relevant regulatory authority.",
        "The manufacturer shall provide the required compliance information to the regulator.",
        "The company must report specified incidents to the competent authority.",
        "Registrants must submit required regulatory information within the prescribed period.",
        "The manufacturer must notify the authority of relevant adverse events.",
        "The company must provide the regulator with required monitoring information.",
        "Reportable pesticide incidents must be communicated to the competent authority.",
        "The registrant must submit the required annual regulatory report.",
        "Companies must notify the regulator when specified events occur.",
        "The manufacturer must provide required compliance reports to the authority.",
    ],
}


def create_examples() -> list[dict]:
    """
    Convert our manually curated regulatory statements
    into the training format.
    """

    examples = []

    for category, statements in TEMPLATES.items():

        for statement in statements:

            examples.append(
                {
                    "instruction": ("Classify the following regulatory statement."),
                    "input": statement,
                    "output": category,
                }
            )

    return examples


def remove_duplicates(examples: list[dict]) -> list[dict]:
    """
    Remove duplicate statements.
    """

    unique_examples = []
    seen = set()

    for example in examples:

        normalized = example["input"].strip().lower()

        if normalized not in seen:
            seen.add(normalized)
            unique_examples.append(example)

    return unique_examples


def stratified_split(
    examples: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split each category independently.

    70% train
    15% validation
    15% test
    """

    grouped = defaultdict(list)

    for example in examples:
        grouped[example["output"]].append(example)

    train = []
    validation = []
    test = []

    for category, category_examples in grouped.items():

        random.shuffle(category_examples)

        total = len(category_examples)

        train_count = int(total * 0.70)
        validation_count = int(total * 0.15)

        category_train = category_examples[:train_count]

        category_validation = category_examples[
            train_count : train_count + validation_count
        ]

        category_test = category_examples[train_count + validation_count :]

        train.extend(category_train)
        validation.extend(category_validation)
        test.extend(category_test)

    random.shuffle(train)
    random.shuffle(validation)
    random.shuffle(test)

    return train, validation, test


def validate_dataset(
    train: list[dict],
    validation: list[dict],
    test: list[dict],
) -> None:

    datasets = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    print("\nRunning dataset validation...")

    # --------------------------------------------------
    # Validate record structure
    # --------------------------------------------------

    required_keys = {
        "instruction",
        "input",
        "output",
    }

    for dataset_name, dataset in datasets.items():

        for index, example in enumerate(dataset):

            if set(example.keys()) != required_keys:
                raise ValueError(f"{dataset_name}[{index}] has invalid fields.")

            if not example["instruction"].strip():
                raise ValueError(f"{dataset_name}[{index}] has empty instruction.")

            if not example["input"].strip():
                raise ValueError(f"{dataset_name}[{index}] has empty input.")

            if example["output"] not in CATEGORIES:
                raise ValueError(
                    f"{dataset_name}[{index}] has invalid category: "
                    f"{example['output']}"
                )

    # --------------------------------------------------
    # Check duplicates inside each split
    # --------------------------------------------------

    split_inputs = {}

    for dataset_name, dataset in datasets.items():

        inputs = [example["input"].strip().lower() for example in dataset]

        if len(inputs) != len(set(inputs)):
            raise ValueError(f"Duplicate statements found inside {dataset_name}.")

        split_inputs[dataset_name] = set(inputs)

    # --------------------------------------------------
    # Check leakage across splits
    # --------------------------------------------------

    train_validation_overlap = split_inputs["train"] & split_inputs["validation"]

    train_test_overlap = split_inputs["train"] & split_inputs["test"]

    validation_test_overlap = split_inputs["validation"] & split_inputs["test"]

    if train_validation_overlap:
        raise ValueError("Data leakage detected between train and validation.")

    if train_test_overlap:
        raise ValueError("Data leakage detected between train and test.")

    if validation_test_overlap:
        raise ValueError("Data leakage detected between validation and test.")

    # --------------------------------------------------
    # Check all categories exist in every split
    # --------------------------------------------------

    for dataset_name, dataset in datasets.items():

        categories = {example["output"] for example in dataset}

        missing = CATEGORIES - categories

        if missing:
            raise ValueError(f"{dataset_name} is missing categories: {missing}")

    print("Dataset validation passed.")


def print_distribution(
    dataset_name: str,
    dataset: list[dict],
) -> None:

    counts = Counter(example["output"] for example in dataset)

    print(f"\n{dataset_name} distribution:")

    for category in sorted(CATEGORIES):
        print(f"  {category:<25} {counts[category]}")


def write_jsonl(
    filename: str,
    examples: list[dict],
) -> None:

    path = DATA_DIR / filename

    with path.open("w", encoding="utf-8") as file:

        for example in examples:

            file.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:

    random.seed(SEED)

    DATA_DIR.mkdir(exist_ok=True)

    # --------------------------------------------------
    # Create dataset
    # --------------------------------------------------

    examples = create_examples()

    print(f"Initial examples: {len(examples)}")

    # --------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------

    examples = remove_duplicates(examples)

    print(f"Unique examples:   {len(examples)}")

    # --------------------------------------------------
    # Stratified split
    # --------------------------------------------------

    train, validation, test = stratified_split(examples)

    # --------------------------------------------------
    # Validate before writing
    # --------------------------------------------------

    validate_dataset(
        train,
        validation,
        test,
    )

    # --------------------------------------------------
    # Write files
    # --------------------------------------------------

    write_jsonl(
        "train.jsonl",
        train,
    )

    write_jsonl(
        "validation.jsonl",
        validation,
    )

    write_jsonl(
        "test.jsonl",
        test,
    )

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    print("\nDataset created successfully.")

    print(f"Total:      {len(examples)}")
    print(f"Train:      {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test:       {len(test)}")

    print_distribution(
        "Train",
        train,
    )

    print_distribution(
        "Validation",
        validation,
    )

    print_distribution(
        "Test",
        test,
    )


if __name__ == "__main__":
    main()
