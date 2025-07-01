from transformers import AutoTokenizer, AutoModelForCausalLM
import os

MODEL_ID = "Qwen/Qwen1.5-0.5B"
MODEL_DIR = "model/qwen-0.5b"

def main():
    if os.path.exists(MODEL_DIR):
        print(f"✅ Model already exists at: {MODEL_DIR}")
    else:
        print("⬇️ Downloading Qwen1.5-0.5B model...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        print(f"✅ Model downloaded and saved to: {MODEL_DIR}")

if __name__ == "__main__":
    main()
