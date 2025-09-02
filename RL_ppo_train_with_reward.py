# file: ppo_train_with_reward.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
from torch.utils.data import DataLoader
import json

from transformers import BitsAndBytesConfig

bnb_cfg = BitsAndBytesConfig(
	load_in_4bit=True,
	bnb_4bit_use_double_quant=True,
	bnb_4bit_quant_type="nf4",
	bnb_4bit_compute_dtype=torch.bfloat16
)


# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------
policy_model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------
# Policy model + reference model with Value Head
# ---------------------------------------------------------
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    policy_model_name,
    quantization_config=bnb_cfg,   # <<--- 4-bit quantization
    #device_map="auto"                 # Spread layers across GPUs if multiple
    device_map={"": 0},
    torch_dtype=torch.bfloat16 # SEE NOTE for this line of code.
)

'''
NOTE: Adding the above line in policy_model resolved the following error:
nan, 'ppo/loss/value': nan, 'ppo/loss/total': nan, 'ppo/policy/entropy': nan, 'ppo/policy/app roxkl': nan, 'ppo/policy/policykl': nan, 'ppo/policy/clipfrac': 0.0, 'ppo/policy/advantages': array([-0.1081972 , -0.10819723, -0.10819727, ..., 0.7 536113 , 1.7679294 , 1.4123744 ], dtype=float32), 'ppo/policy/advantages_mean': -8.847564458847046e-09, 'ppo/policy/ratio': array([ 1., 1., 1., ..., nan, nan, nan], dtype=float32), 'ppo/returns/mean': -1.3111425638198853, 'ppo/returns/var': 0.5395309925079346, 'ppo/val/vpred': nan, 'ppo/val/error': nan, 'ppo/val/clipfrac': 0.0, 'ppo/val/mean': -1.5473783016204834, 'ppo/val/var': 4.368826866149902, 'ppo/val/var_explained': nan, 'ppo/learning_rate': 1e-05, 'time/ppo/forward_pass': 0.512310266494751, 'time/ppo/compute_rewards': 0.0004951953887939453, 'time/ppo/compute_advantages': 0.03375387191772461, 'time/ppo/optimize_step': 2.2348992824554443, 'time/ppo/calc_stats': 0.035600900650024414, 'time/ppo/total': 2.8172545433044434} /pytorch/aten/src/ATen/native/cuda/TensorCompare.cu:112: _assert_async_cuda_kernel: block: [0,0,0], thread: [0,0,0] Assertion probability tensor contains either inf, nan or element < 0 failed.

torch_dtype=torch.bfloat16 ensures that all weights and activations are
computed in bf16 (except the 4-bit quantized weights, which remain quantized).
This keeps gradients and logits stable during PPO updates.
'''

ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    policy_model_name,
    quantization_config=bnb_cfg,
    #device_map="auto"
    device_map={"": 0}
)

print("✅ Loaded Mistral 7B Instruct quantized in 4-bit.")

'''
# ---------------------------------------------------------
# 1. Policy Model (Causal LM) with Value Head
# ---------------------------------------------------------
policy_model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # replace with your desired base model
#PLUGIN-Switch the base model here.
tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(policy_model_name)
'''

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ref_model.to(device)
policy_model.to(device)

# ---------------------------------------------------------
# 2. Reward Model (SequenceClassification)
# ---------------------------------------------------------
reward_model_name = "output/reward_model_pairwise"
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name)
reward_model.to(device)
num_labels = reward_model.config.num_labels
print(f"[INFO] Reward model outputs {num_labels} label(s).")

# ---------------------------------------------------------
# 3. PPO Trainer
# ---------------------------------------------------------

# Monitoring with Weights & Biases
import wandb

wandb.init(
    project="ppo-mistral-reward-tuning",
    config={
        "policy_model": policy_model_name,
        "reward_model": reward_model_name,
        "batch_size": 2,
        "learning_rate": 1e-5,
        "quantization": "4-bit-nf4",
    }
)

ppo_config = PPOConfig(
    batch_size=2,
    mini_batch_size=1,
    learning_rate=1e-5,
    log_with="wandb"
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=ref_model,
    tokenizer=tokenizer
)

# ---------------------------------------------------------
# 4. Get Prompts 
# ---------------------------------------------------------
prompts = []
with open("output/pairwise_prefs_part_1.jsonl", "r") as f:
    for line in f:
        obj = json.loads(line)
        prompts.append(obj["prompt"])

print(f"Extracted {len(prompts)} prompts")

# ---------------------------------------------------------
# 5. PPO Training Loop (with batching)
# ---------------------------------------------------------
loader = DataLoader(prompts, batch_size=ppo_config.batch_size, shuffle=True)

for step, prompt_batch in enumerate(loader):
    if step >= 10:  # demo: just 10 steps
        break

    # Guidance on num of steps to use:
    # https://chatgpt.com/share/68b6d7dd-2664-8002-babf-96158c1874cc

    # Tokenize this batch of prompts
    batch = reward_tokenizer(prompt_batch, return_tensors="pt", padding=True, truncation=True)
    batch = {k: v.to(device) for k, v in batch.items()}

    # Generate responses from policy
    response_ids = policy_model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        max_new_tokens=50,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True
    )

    responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_ids]

    # Compute rewards from reward model
    with torch.no_grad():
        reward_inputs = reward_tokenizer(responses, return_tensors="pt", padding=True, truncation=True).to(device)
        reward_logits = reward_model(**reward_inputs).logits

        if num_labels == 1:
            rewards = reward_logits.squeeze(-1)
        else:
            if num_labels == 2:
                rewards = torch.softmax(reward_logits, dim=-1)[:, 1]
            else:
                rewards = reward_logits.max(dim=-1).values

    # Convert all to lists of tensors for PPO
    queries = [batch["input_ids"][i] for i in range(batch["input_ids"].size(0))]
    responses_list = [response_ids[i] for i in range(response_ids.size(0))]
    rewards_list = [rewards[i] for i in range(rewards.size(0))]

    # PPO step
    stats = ppo_trainer.step(queries, responses_list, rewards_list)

    print(f"[STEP {step}] Rewards: {rewards.tolist()} | PPO Stats: {stats}")

    # log metrics to W&B
    wandb.log({
        "step": step,
        "reward_mean": torch.mean(rewards).item(),
        "reward_std": torch.std(rewards).item(),
        "ppo/value_loss": stats["ppo/loss/value"],
        "ppo/policy_loss": stats["ppo/loss/total"],
        "ppo/kl": stats["ppo/policy/policykl"],
        "ppo/entropy": stats["ppo/policy/entropy"],
    })

    # optional: log sample generations
    wandb.log({"samples": wandb.Table(columns=["prompt", "response"], 
        data=[[prompt_batch[i], responses[i]] for i in range(len(responses))])})


#PLUGIN-- save the model
print("✅ PPO demo finished")

