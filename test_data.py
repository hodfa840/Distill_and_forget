import json
import random

# Load and preview training dataset
with open("data/train_with_canaries.jsonl", "r") as f:
    train_data = [json.loads(line) for line in f]

# Load and preview canary dataset
with open("data/canary_eval_set.jsonl", "r") as f:
    canary_data = [json.loads(line) for line in f]

# Print 3 random examples from each
print("\n=== Train Sample ===")
for sample in random.sample(train_data, 3):
    print(sample)

print("\n=== Canary Sample ===")
for sample in random.sample(canary_data, 3):
    print(sample)
