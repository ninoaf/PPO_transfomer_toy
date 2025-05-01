# Load Fine-Tuned Full Reward Model (Corrected)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Load the saved full fine-tuned reward model
reward_model_path = "./lora_reward_model-final"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(reward_model_path)

# Load full model
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path,
    num_labels=1  # Regression output
)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reward_model = reward_model.to(device)
reward_model.eval()

# 2. Define scoring function
def score_response(prompt, response, model, tokenizer):
    full_text = prompt + "\n" + response
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits.squeeze().item()
    return score

# 3. Example Testing
if __name__ == "__main__":
    prompt = "Explain the theory of relativity."
    good_response = "Let's think step-by-step. Relativity consists of two theories by Einstein: special and general relativity..."
    bad_response = "It's something about fast cars or something."

    good_score = score_response(prompt, good_response, reward_model, tokenizer)
    bad_score = score_response(prompt, bad_response, reward_model, tokenizer)

    print(f"✅ Good Response Score: {good_score:.4f}")
    print(f"❌ Bad Response Score: {bad_score:.4f}")
