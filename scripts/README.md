# HPC Scripts

This directory contains SLURM batch scripts for running fine-tuning and evaluation jobs on GPU clusters.

## Scripts

| Script | Purpose | GPU / Time |
|--------|---------|-----------|
| `run_finetune_embeddinggemma.sbatch` | Fine-tune Google EmbeddingGemma-300M on Amharic retrieval | H100, 48 h |
| `run_finetune_harrier.sbatch` | Fine-tune Microsoft Harrier OSS 270M on Amharic retrieval | H100, 48 h |
| `run_evaluate_gemma.sbatch` | Evaluate zero-shot and fine-tuned EmbeddingGemma | H100, 4 h |
| `run_evaluate_harrier.sbatch` | Evaluate zero-shot and fine-tuned Harrier | H100, 4 h |

## Setup

**1. Edit the configuration block at the top of each script** before submitting:

```bash
REPO_DIR="/absolute/path/to/amharic-neural-ir"   # path to repo root
CONDA_ENV="/absolute/path/to/conda-envs/amharicir" # conda env from amharicir-environment.yml
export WANDB_ENTITY="your-wandb-entity"            # W&B username or org (fine-tuning scripts only)
```

Or set them as environment variables before calling `sbatch`:

```bash
export REPO_DIR=/home/you/amharic-neural-ir
export CONDA_ENV=/home/you/envs/amharicir
export WANDB_ENTITY=your-team
sbatch scripts/run_finetune_embeddinggemma.sbatch
```

**2. Create the conda environment** (once):

```bash
conda env create -f amharicir-environment.yml
```

**3. Authenticate** with Hugging Face and Weights & Biases (once per cluster):

```bash
huggingface-cli login
wandb login
```

**4. Submit** from the repository root:

```bash
sbatch scripts/run_finetune_embeddinggemma.sbatch
sbatch scripts/run_finetune_harrier.sbatch
```

Logs are written to `logs-slurm/` in the repository root.

## Adapting to other schedulers

The scripts use SLURM directives (`#SBATCH`). For PBS/Torque or other schedulers, replace the `#SBATCH` header lines with the equivalent directives and adjust the `$SLURM_JOB_ID` variable references accordingly.
