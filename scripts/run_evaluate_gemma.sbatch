#!/bin/bash -l
#SBATCH --job-name=eval_gemma
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --output=logs-slurm/eval_gemma_bf16fix_%j.out
#SBATCH --error=logs-slurm/eval_gemma_bf16fix_%j.err

set -eo pipefail

# ── Configure before submitting ───────────────────────────────────────────────
REPO_DIR="${REPO_DIR:-/path/to/amharic-neural-ir}"
CONDA_ENV="${CONDA_ENV:-/path/to/conda-envs/amharicir}"
# ─────────────────────────────────────────────────────────────────────────────

cd "$REPO_DIR"
mkdir -p logs-slurm

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

export PATH="${CONDA_ENV}/bin:${PATH}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
unset PYTHONPATH

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_TOKEN="$(hf auth token 2>/dev/null)"

set -u

echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
nvidia-smi

echo ""
echo "###############################################################"
echo "# Gemma Embed 300M — ZERO-SHOT (google/embeddinggemma-300m)  #"
echo "###############################################################"
python -u evaluate_ir.py \
  --model_id "google/embeddinggemma-300m" \
  --eval_batch_size 32 \
  --max_seq_length 512 \
  --matryoshka_dims 768 256

echo ""
echo "###############################################################"
echo "# Gemma Embed 300M — FINE-TUNED BF16 (embeddinggemma-300m-am-finetuned-bf16fix) #"
echo "###############################################################"
python -u evaluate_ir.py \
  --model_id "embeddinggemma-300m-am-finetuned-bf16fix" \
  --eval_batch_size 32 \
  --max_seq_length 512 \
  --matryoshka_dims 768 256

echo "Job finished on: $(date)"
