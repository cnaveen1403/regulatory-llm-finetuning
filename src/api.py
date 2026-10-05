from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# MODEL_PATH = Path("models/regulatory-qwen-lora-classifier-v1")
MODEL_PATH = Path("models/production/regulatory-qwen-lora-classifier-v1")

LABELS = [
    "SAFETY_REQUIREMENT",
    "LABELING_REQUIREMENT",
    "REGISTRATION_REQUIREMENT",
    "STORAGE_REQUIREMENT",
    "APPLICATION_REQUIREMENT",
    "ENVIRONMENTAL_REQUIREMENT",
    "REPORTING_REQUIREMENT",
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()

print(f"Using device: {DEVICE}")
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading base model...")

base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(LABELS),
)

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    MODEL_PATH,
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


app = FastAPI(
    title="Regulatory Classification API",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Regulatory statement to classify",
    )


class PredictionResponse(BaseModel):
    text: str
    label: str
    confidence: float


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": BASE_MODEL,
        "device": str(DEVICE),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    probabilities = torch.softmax(logits, dim=-1)

    predicted_id = torch.argmax(
        probabilities,
        dim=-1,
    ).item()

    confidence = probabilities[0, predicted_id].item()

    return PredictionResponse(
        text=text,
        label=LABELS[predicted_id],
        confidence=round(confidence, 4),
    )
