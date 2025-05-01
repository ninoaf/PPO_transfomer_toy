# LoRA Fine-Tuning for Reward Model (RoBERTa)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, concatenate_datasets

# Load preference dataset
dataset = load_dataset("Dahoas/full-hh-rlhf")

# Preprocess
def preprocess_data(example):
    return {"text": example["chosen"], "label": float(1)}

def preprocess_rejected(example):
    return {"text": example["rejected"], "label": float(0)}


chosen_dataset = dataset["train"].map(preprocess_data)
rejected_dataset = dataset["train"].map(preprocess_rejected)

# Correctly merge them
reward_dataset = concatenate_datasets([chosen_dataset, rejected_dataset])

# Load base model and tokenizer
base_model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1)

# Apply LoRA to model
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "key", "value", "dense", "intermediate.dense", "output.dense"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

reward_dataset = reward_dataset.map(tokenize_function, batched=True)
reward_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# Print dataset size
print(f"Dataset size: {len(reward_dataset)} samples")
# Shuffle and subsample
#reward_dataset = reward_dataset.shuffle(seed=42).select(range(5000))


# Define training arguments
training_args = TrainingArguments(
    output_dir="./lora_reward_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch",
    report_to="none",
    fp16=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=reward_dataset,
)

# Train model
trainer.train()

# Save final adapter
model.save_pretrained("./lora_reward_model-final")
tokenizer.save_pretrained("./lora_reward_model-final")
