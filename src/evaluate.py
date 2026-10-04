import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from inference import load_model, load_test_data, predict


VALID_LABELS = [
    "SAFETY_REQUIREMENT",
    "LABELING_REQUIREMENT",
    "REGISTRATION_REQUIREMENT",
    "STORAGE_REQUIREMENT",
    "APPLICATION_REQUIREMENT",
    "ENVIRONMENTAL_REQUIREMENT",
    "REPORTING_REQUIREMENT",
]


def normalize_prediction(prediction: str) -> str:
    """
    Normalize model output so formatting differences
    don't count as semantic differences.

    Example:
        ENVIRONMENTAL REQUIREMENT
        -> ENVIRONMENTAL_REQUIREMENT
    """

    prediction = prediction.strip().upper()

    prediction = prediction.replace(" ", "_")

    return prediction


def evaluate_model():

    tokenizer, model = load_model()

    examples = load_test_data()

    expected_labels = []
    predictions = []

    print(
        f"\nEvaluating LoRA model on "
        f"{len(examples)} test examples...\n"
    )

    for index, example in enumerate(
        examples,
        start=1,
    ):

        prediction = predict(
            tokenizer,
            model,
            example["input"],
        )

        normalized_prediction = normalize_prediction(
            prediction
        )

        expected = example["output"]

        expected_labels.append(expected)
        predictions.append(normalized_prediction)

        print("=" * 70)

        print(f"Example:    {index}")
        print(f"Expected:   {expected}")
        print(f"Predicted:  {prediction}")
        print(f"Normalized: {normalized_prediction}")

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    accuracy = accuracy_score(
        expected_labels,
        predictions,
    )

    print(
        f"\nAccuracy: {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    invalid_predictions = [
        prediction
        for prediction in predictions
        if prediction not in VALID_LABELS
    ]

    print(
        f"Invalid predictions: "
        f"{len(invalid_predictions)} / {len(predictions)}"
    )

    if invalid_predictions:
        print("\nInvalid labels:")

        for label in sorted(set(invalid_predictions)):
            print(f"  - {label}")

    print("\nClassification Report:")

    print(
        classification_report(
            expected_labels,
            predictions,
            labels=VALID_LABELS,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")

    matrix = confusion_matrix(
        expected_labels,
        predictions,
        labels=VALID_LABELS,
    )

    print()

    print("Labels:")
    for index, label in enumerate(VALID_LABELS):
        print(f"{index}: {label}")

    print("\nMatrix:")
    print(matrix)


if __name__ == "__main__":
    evaluate_model()