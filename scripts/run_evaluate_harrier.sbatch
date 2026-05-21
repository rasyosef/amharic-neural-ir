#!/bin/bash -l
#SBATCH --job-name=eval_harrier
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --output=logs-slurm/eval_harrier_prompted%j.out
#SBATCH --error=logs-slurm/eval_harrier_prompted%j.err

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
echo "# Harrier OSS 270M — ZERO-SHOT (microsoft/harrier-oss-v1-270m) #"
echo "###############################################################"
python -u evaluate_ir.py \
  --model_id "microsoft/harrier-oss-v1-270m" \
  --eval_batch_size 32 \
  --max_seq_length 512 \
  --matryoshka_dims 640 256 \
  --query_prompt "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "

echo ""
echo "###############################################################"
echo "# Harrier OSS 270M — FINE-TUNED (harrier-270m-am-finetuned)  #"
echo "###############################################################"
python -u evaluate_ir.py \
  --model_id "harrier-270m-am-finetuned" \
  --eval_batch_size 32 \
  --max_seq_length 512 \
  --matryoshka_dims 640 256

echo "Job finished on: $(date)"
