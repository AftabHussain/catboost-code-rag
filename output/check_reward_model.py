import os
import torch
from transformers import AutoConfig

model_folder = "reward_model_pairwise"

# List files in the folder
print("Files in model folder:", os.listdir(model_folder))

# Try to load as a PyTorch checkpoint
try:
    checkpoint = torch.load(os.path.join(model_folder, "pytorch_model.bin"), map_location="cpu")
    if isinstance(checkpoint, dict):
        print("Looks like a PyTorch model!")
except Exception as e:
    print("Not a plain PyTorch checkpoint:", e)

# Try to load as a Hugging Face transformers model
try:
    config = AutoConfig.from_pretrained(model_folder)
    print("Looks like a Hugging Face Transformers model!")
except Exception as e:
    print("Not a Hugging Face Transformers model:", e)

