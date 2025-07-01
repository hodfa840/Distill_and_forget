#!/bin/bash
#SBATCH --job-name=lora_qwen
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# === Activate conda environment ===
eval "$(conda shell.bash hook)"
conda activate /proj/berzelius-aiics-real/users/x_hodfa/nlp1

# === Navigate to project folder ===
cd /proj/berzelius-aiics-real/users/x_hodfa/Distill_and_forget

# === Make sure log folder exists ===
mkdir -p log

# === Run model download script ===
python script/check_and_download_model.py

# === Run training ===
python script/train_lora_qwen.py
