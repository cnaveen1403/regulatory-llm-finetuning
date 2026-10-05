import torch
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = Path("models/regulatory-qwen-lora-classifier-v1")


LABELS = [
    "SAFETY_REQUIREMENT",
    "LABELING_REQUIREMENT",
    "REGISTRATION_REQUIREMENT",
    "STORAGE_REQUIREMENT",
    "APPLICATION_REQUIREMENT",
    "ENVIRONMENTAL_REQUIREMENT",
    "REPORTING_REQUIREMENT",
]


def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_id = logits.argmax(dim=-1).item()

    return LABELS[predicted_id]


if __name__ == "__main__":
    tokenizer, model, device = load_model()

    text = "The pesticide must be stored separately from consumable products."

    prediction = predict(
        text,
        tokenizer,
        model,
        device,
    )

    print(f"\nInput: {text}")
    print(f"Prediction: {prediction}")
