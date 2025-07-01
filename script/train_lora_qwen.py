import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# === CONFIGURATION ===
model_id = "model/qwen-0.5b"  # Local model path
output_dir = "data/qwen_canary_lora"
train_path = "data/train_with_canaries.jsonl"

# === LOAD DATASET ===
with open(train_path, "r") as f:
    train_data = [json.loads(line) for line in f]
train_dataset = Dataset.from_list(train_data)

# === TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# === LOAD MODEL IN 4-BIT ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=bnb_config
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# === LoRA CONFIG ===
def find_lora_targets(model):
    target_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            target_modules.add(name.split(".")[-1])
    return list(sorted(target_modules))

target_modules = find_lora_targets(model)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=target_modules
)

model = get_peft_model(model, peft_config)

# === PROMPT FORMATTING ===
def tokenize(example):
    prompt = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    )
    return tokenizer(prompt, padding="max_length", truncation=True, max_length=512)

# === TOKENIZE DATA ===
tokenized_dataset = train_dataset.map(
    tokenize,
    remove_columns=train_dataset.column_names,
    num_proc=4,
    desc="Tokenizing"
)

# === TRAINING ARGS ===
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    fp16=True,
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    remove_unused_columns=False,
    run_name="lora-qwen-training"
)

# === TRAIN ===
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    args=args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# === SAVE MODEL ===
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ LoRA fine-tuned model saved at: {output_dir}")
