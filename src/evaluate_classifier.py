import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = Path("models/regulatory-qwen-classifier-v1")
TEST_FILE = Path("data/test.jsonl")


LABELS = [
    "SAFETY_REQUIREMENT",
    "LABELING_REQUIREMENT",
    "REGISTRATION_REQUIREMENT",
    "STORAGE_REQUIREMENT",
    "APPLICATION_REQUIREMENT",
    "ENVIRONMENTAL_REQUIREMENT",
    "REPORTING_REQUIREMENT",
]


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def main():

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading classification model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    model.to(device)
    model.eval()

    test_records = load_jsonl(TEST_FILE)

    print(f"Test examples: {len(test_records)}")

    y_true = []
    y_pred = []

    print("\nPredictions:\n")

    with torch.no_grad():

        for index, record in enumerate(test_records, start=1):

            text = record["input"]
            actual = record["output"]

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )

            inputs = {key: value.to(device) for key, value in inputs.items()}

            outputs = model(**inputs)

            predicted_id = torch.argmax(outputs.logits, dim=-1).item()

            predicted = LABELS[predicted_id]

            y_true.append(actual)
            y_pred.append(predicted)

            status = "✓" if actual == predicted else "✗"

            print(
                f"{index:2d}. {status} "
                f"Actual: {actual:<28} "
                f"Predicted: {predicted}"
            )

    accuracy = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 70)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("=" * 70)

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            target_names=LABELS,
            zero_division=0,
        )
    )

    print("Confusion Matrix:\n")

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    print("Actual \\ Predicted")
    print(" " * 28 + " ".join(f"{i:4d}" for i in range(len(LABELS))))

    for i, label in enumerate(LABELS):
        print(f"{label:<28} " + " ".join(f"{value:4d}" for value in matrix[i]))


if __name__ == "__main__":
    main()
