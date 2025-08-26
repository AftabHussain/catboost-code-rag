'''
2) Train a pairwise Reward Model (RM) on those preferences

This uses a pairwise loss: -log σ(r_chosen − r_rejected).
'''

# file: train_reward_model_pairwise.py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import json
import os
import pandas as pd

# ------------------------
# Config
# ------------------------
RM_MODEL_NAME = "bert-base-uncased"          # base model for reward model
DATA_PATH = "output/pairwise_prefs_part_1_test.jsonl"   # JSONL: {"prompt": ..., "chosen": ..., "rejected": ...}
OUTPUT_DIR = "output/reward_model_pairwise"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 5
LR = 5e-5
MAX_LENGTH = 256
VAL_SPLIT = 0.1  # fraction for validation
CSV_LOG = os.path.join(OUTPUT_DIR, "training_log.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------
# Dataset
# ------------------------
class PairwiseDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=256):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        chosen = item["chosen"]
        rejected = item["rejected"]

        chosen_enc = self.tokenizer(
            chosen, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt"
        )
        rejected_enc = self.tokenizer(
            rejected, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt"
        )
        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }

# ------------------------
# Load tokenizer and model
# ------------------------
tokenizer = AutoTokenizer.from_pretrained(RM_MODEL_NAME)
rm_model = AutoModelForSequenceClassification.from_pretrained(RM_MODEL_NAME, num_labels=1)
rm_model.to(DEVICE)

# ------------------------
# DataLoader with validation split
# ------------------------
dataset = PairwiseDataset(DATA_PATH, tokenizer, MAX_LENGTH)
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------
# Optimizer
# ------------------------
optimizer = AdamW(rm_model.parameters(), lr=LR)

# ------------------------
# Training loop with validation
# ------------------------
log_data = []

print("🚀 Starting Reward Model training...")
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    rm_model.train()
    epoch_loss = 0.0

    for batch in tqdm(train_loader, desc="Training batches", leave=False):
        optimizer.zero_grad()

        # Move to device
        chosen_ids = batch["chosen_input_ids"].to(DEVICE)
        chosen_mask = batch["chosen_attention_mask"].to(DEVICE)
        rejected_ids = batch["rejected_input_ids"].to(DEVICE)
        rejected_mask = batch["rejected_attention_mask"].to(DEVICE)

        # Forward pass
        chosen_scores = rm_model(input_ids=chosen_ids, attention_mask=chosen_mask).logits.squeeze(-1)
        rejected_scores = rm_model(input_ids=rejected_ids, attention_mask=rejected_mask).logits.squeeze(-1)

        # Pairwise loss: maximize chosen - rejected
        loss = -torch.nn.functional.logsigmoid(chosen_scores - rejected_scores).mean()

        # Backward
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    print(f"  Average training loss: {avg_train_loss:.4f}")

    # ------------------------
    # Validation
    # ------------------------
    rm_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation batches", leave=False):
            chosen_ids = batch["chosen_input_ids"].to(DEVICE)
            chosen_mask = batch["chosen_attention_mask"].to(DEVICE)
            rejected_ids = batch["rejected_input_ids"].to(DEVICE)
            rejected_mask = batch["rejected_attention_mask"].to(DEVICE)

            chosen_scores = rm_model(input_ids=chosen_ids, attention_mask=chosen_mask).logits.squeeze(-1)
            rejected_scores = rm_model(input_ids=rejected_ids, attention_mask=rejected_mask).logits.squeeze(-1)

            correct += (chosen_scores > rejected_scores).sum().item()
            total += chosen_scores.size(0)

    val_accuracy = correct / total
    print(f"  Validation accuracy: {val_accuracy:.4f}")

    # ------------------------
    # Save checkpoint each epoch
    checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch{epoch+1}")
    rm_model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    # ------------------------
    # Log to CSV
    log_data.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_accuracy": val_accuracy
    })
    pd.DataFrame(log_data).to_csv(CSV_LOG, index=False)

# ------------------------
# Final save
# ------------------------
rm_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n✅ Reward Model training complete. Saved to {OUTPUT_DIR}")
print(f"📄 Training log saved to {CSV_LOG}")

