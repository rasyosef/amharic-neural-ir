#!/bin/bash -l
#SBATCH --job-name=gemma_am_embed
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH -c 16
#SBATCH --output=logs-slurm/gemma_am_embed_%j.out
#SBATCH --error=logs-slurm/gemma_am_embed_%j.err

set -eo pipefail

# ── Configure before submitting ───────────────────────────────────────────────
# REPO_DIR: absolute path to the cloned repository root
REPO_DIR="${REPO_DIR:-/path/to/amharic-neural-ir}"
# CONDA_ENV: path to the conda environment created from amharicir-environment.yml
CONDA_ENV="${CONDA_ENV:-/path/to/conda-envs/amharicir}"
# WANDB_ENTITY: your Weights & Biases username or organization
export WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-entity}"
# ─────────────────────────────────────────────────────────────────────────────

cd "$REPO_DIR"
mkdir -p logs-slurm

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

export PATH="${CONDA_ENV}/bin:${PATH}"

# Prevent Python from mixing ~/.local packages with this conda env
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
unset PYTHONPATH

# Optional: helps with CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hugging Face auth/cache
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_TOKEN="$(hf auth token 2>/dev/null)"

# Weights & Biases logging
export WANDB_PROJECT=amharic-embeddinggemma
export WANDB_RUN_GROUP=embeddinggemma-300m-amharic-bf16fix
export WANDB_RUN_NAME="embeddinggemma-300m-amharic-bf16fix-${SLURM_JOB_ID}"

export WANDB_DIR="${WANDB_DIR:-${HOME}/wandb}"
mkdir -p "$WANDB_DIR"

set -u

echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"

echo "Python executable:"
which python

echo "Python package versions:"
python - <<'PY'
import sentence_transformers
import torch
import transformers

print("sentence_transformers:", sentence_transformers.__version__)
print("transformers:", transformers.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
PY

echo "Checking Hugging Face model access:"
python - <<'PY'
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="google/embeddinggemma-300m",
    filename="config.json",
    token=True,
)
print("HF access OK:", path)
PY

echo "Checking Weights & Biases login:"
wandb status

nvidia-smi

python -u finetune_embeddinggemma_amharic.py \
  --dataset_name "yosefw/amharic-news-retrieval-dataset-v2-with-negatives-V2" \
  --model_id "google/embeddinggemma-300m" \
  --output_dir "embeddinggemma-300m-am-finetuned-bf16fix" \
  --num_train_epochs 6 \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --learning_rate 4e-5 \
  --max_seq_length 512 \
  --logging_steps 50 \
  --report_to wandb \
  --run_name "embeddinggemma-300m-amharic-bf16fix-${SLURM_JOB_ID}"

echo "Job finished on: $(date)"
