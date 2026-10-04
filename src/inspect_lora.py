from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


def main():
    print("Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    total, trainable = count_parameters(model=model)

    print("\nBefore LoRA")
    print("-" * 40)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    total, trainable = count_parameters(model)

    print("\nAfter LoRA")
    print("-" * 40)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    percentage = 100 * trainable / total

    print(f"Trainable percentage:  {percentage:.4f}%")

    print("\nLoRA configuration")
    print("-" * 40)
    print(f"Rank (r):             {lora_config.r}")
    print(f"Alpha:                {lora_config.lora_alpha}")
    print(f"Target modules:       {lora_config.target_modules}")
    print(f"Dropout:              {lora_config.lora_dropout}")

    print("\nTrainable parameters:")
    print("-" * 40)

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(name, parameter.shape)


if __name__ == "__main__":
    main()
