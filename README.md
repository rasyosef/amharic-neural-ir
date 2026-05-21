

# The Multilingual Curse at the Retrieval Layer: Evidence from Amharic

[![ACL 2026](https://img.shields.io/badge/ACL-2026-blue)](https://2026.aclweb.org/)
[![MeLLM Workshop](https://img.shields.io/badge/Workshop-MeLLM-informational)](https://mellm.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/collections/rasyosef/amharic-neural-ir-models)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/collections/kiyam/amharic-fine-tuned-multilingual-retrievers)
[![License](https://img.shields.io/badge/license-LICENSE-lightgrey)](LICENSE)

This repository accompanies the ACL 2026 MeLLM Workshop paper **”The Multilingual Curse at the Retrieval Layer: Evidence from Amharic.”** It provides notebook-based training and evaluation workflows for dense retrieval, late interaction (ColBERT-style), sparse retrieval (SPLADE-style), and cross-encoder reranking in Amharic.

**Core artifacts**
- **Benchmark**: Amharic Passage Retrieval Dataset V2 with a fixed 90/10 train–test split (68,000 query–passage pairs).
- **Model suite**: Amharic-specific checkpoints spanning dense bi-encoders, late-interaction (ColBERT-style), learned sparse retrievers (SPLADE-style), and cross-encoder rerankers.
- **Workflows**: notebook implementations for preprocessing, training, and evaluation.

**Hugging Face resources**
- Dataset: [rasyosef/Amharic-Passage-Retrieval-Dataset-V2](https://huggingface.co/datasets/rasyosef/Amharic-Passage-Retrieval-Dataset-V2)
- Monolingual Amharic models: [rasyosef/amharic-neural-ir-models](https://huggingface.co/collections/rasyosef/amharic-neural-ir-models)
- Fine-tuned multilingual models (this paper): [kiyam/amharic-fine-tuned-multilingual-retrievers](https://huggingface.co/collections/kiyam/amharic-fine-tuned-multilingual-retrievers)

**Monolingual Amharic models**
- [rasyosef/RoBERTa-Amharic-Embed-Base](https://huggingface.co/rasyosef/RoBERTa-Amharic-Embed-Base)
- [rasyosef/RoBERTa-Amharic-Embed-Medium](https://huggingface.co/rasyosef/RoBERTa-Amharic-Embed-Medium)
- [rasyosef/ColBERT-Amharic-Base](https://huggingface.co/rasyosef/ColBERT-Amharic-Base)
- [rasyosef/ColBERT-Amharic-Medium](https://huggingface.co/rasyosef/ColBERT-Amharic-Medium)
- [rasyosef/SPLADE-RoBERTa-Amharic-Base](https://huggingface.co/rasyosef/SPLADE-RoBERTa-Amharic-Base)
- [rasyosef/SPLADE-RoBERTa-Amharic-Medium](https://huggingface.co/rasyosef/SPLADE-RoBERTa-Amharic-Medium)
- [rasyosef/RoBERTa-Amharic-Reranker-Base](https://huggingface.co/rasyosef/RoBERTa-Amharic-Reranker-Base)
- [rasyosef/RoBERTa-Amharic-Reranker-Medium](https://huggingface.co/rasyosef/RoBERTa-Amharic-Reranker-Medium)

**Amharic-fine-tuned multilingual models**
- [kiyam/EmbeddingGemma-300M-Amharic](https://huggingface.co/kiyam/EmbeddingGemma-300M-Amharic) — MRR@10: 0.718, NDCG@10: 0.753
- [kiyam/Harrier-270M-Amharic](https://huggingface.co/kiyam/Harrier-270M-Amharic) — MRR@10: 0.760, NDCG@10: 0.795


## Notebook-first workflow

This codebase is organized primarily as Jupyter notebooks (rather than standalone `.py` scripts). The goal is to keep the full pipeline easy to follow and modify step-by-step, especially for practitioners. Because the dataset used in these workflows is relatively small, we keep the main experiments and analysis in notebook format for clarity and quick iteration.


**Practical notes**

* Run notebooks **from the repository root** so relative paths resolve correctly.
* Each notebook is intended to be runnable end-to-end, in the order described below.
* If you prefer scripts, you can export notebooks with:
  * `jupyter nbconvert --to script <notebook-path>.ipynb`

## Quickstart

### Recommended (Conda, GPU-friendly)

Create a conda environment from `amharicir-environment.yml`:

```bash
conda env create -f amharicir-environment.yml
conda activate amharicir
jupyter lab
```

Then open one of:

* `evaluation/evaluate-amharic-embedding-passage-retrieval.ipynb`
* `evaluation/evaluate-amharic-colbert-passage-retrieval.ipynb`
* `evaluation/evaluate-amharic-splade-passage-retrieval.ipynb`
* `evaluation/evaluate-amharic-rerankers-passage-retrieval.ipynb`

### Optional (venv + requirements.txt, pip-only)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jupyter
jupyter lab
```

Then open one of:

* `evaluation/evaluate-amharic-embedding-passage-retrieval.ipynb`
* `evaluation/evaluate-amharic-colbert-passage-retrieval.ipynb`
* `evaluation/evaluate-amharic-splade-passage-retrieval.ipynb`
* `evaluation/evaluate-amharic-rerankers-passage-retrieval.ipynb`

## Installation

Canonical environment path for this repo:
- `conda` with `amharicir-environment.yml` (Python 3.10)

```bash
conda env create -f amharicir-environment.yml
conda activate amharicir
```

For pip-only workflows, use `requirements.txt` with a virtual environment.

Python version:
- The environment file pins Python to `3.10`.
- Notebook metadata in this repo includes multiple runtime versions (`3.10.12`, `3.11`, `3.12.12`), so exact results may vary across runtimes.
- Dependencies are pinned in both `amharicir-environment.yml` and `requirements.txt`.

## Usage

### 1) Evaluate pretrained retrieval models

Dense embedding retrieval:
```bash
jupyter lab evaluation/evaluate-amharic-embedding-passage-retrieval.ipynb
```

ColBERT-style retrieval:
```bash
jupyter lab evaluation/evaluate-amharic-colbert-passage-retrieval.ipynb
```

SPLADE-style retrieval:
```bash
jupyter lab evaluation/evaluate-amharic-splade-passage-retrieval.ipynb
```

Two-stage retrieval + reranking:
```bash
jupyter lab evaluation/evaluate-amharic-rerankers-passage-retrieval.ipynb
```

### 2) Preprocess / hard-negative mining

```bash
jupyter lab preprocessing/hard-negatives-mining-amharic-retrieval-dataset.ipynb
```

### 3) Train models

Embeddings:
- `training/embeddings-amharic/train-roberta-amharic-embed-base.ipynb`
- `training/embeddings-amharic/train-roberta-amharic-embed-medium.ipynb`

ColBERT:
- `training/colbert-amharic/train-colbert-amharic-base.ipynb`
- `training/colbert-amharic/train-colbert-amharic-medium.ipynb`

SPLADE:
- `training/splade-amharic/train-splade-roberta-amharic-base.ipynb`
- `training/splade-amharic/train-splade-roberta-amharic-medium.ipynb`

Cross-encoder reranker:
- `training/crossencoder-amharic/train-roberta-amharic-reranker-base.ipynb`
- `training/crossencoder-amharic/train-roberta-amharic-reranker-medium.ipynb`

### 4) HPC / SLURM scripts

For running fine-tuning or evaluation on a GPU cluster via SLURM, see [`scripts/`](scripts/). Each script has a short configuration block at the top — set `REPO_DIR`, `CONDA_ENV`, and (for training) `WANDB_ENTITY` before submitting:

```bash
sbatch scripts/run_finetune_embeddinggemma.sbatch
sbatch scripts/run_finetune_harrier.sbatch
sbatch scripts/run_evaluate_gemma.sbatch
sbatch scripts/run_evaluate_harrier.sbatch
```

Logs are written to `logs-slurm/`.

## Reproducibility

### Data contract (as used in notebooks)
- Evaluation notebooks load: `rasyosef/Amharic-Passage-Retrieval-Dataset-V2`
- Training/preprocessing notebooks load: `yosefw/amharic-news-retrieval-dataset-v2-with-negatives-V2` or `rasyosef/amharic-passage-retrieval-dataset-v2`
- Common ID fields in workflows: `query_id`, `passage_id`

### Seeds
- Multiple training notebooks use `seed=42` (for example in dataset shuffling and training arguments).
- A few evaluation paths are notebook-interactive and rely on per-cell execution order; rerunning from top to bottom is recommended.

### Runtime and hardware notes
- Experiments in this repository were run in GPU-backed environments, ( A100 and T4 GPUs).
- Some code paths explicitly set `device="cuda"` or `device="cuda" if torch.cuda.is_available() else "cpu"`.
- A recorded run in `evaluation/evaluate-amharic-colbert-passage-retrieval.ipynb` shows a corpus chunk stage of about `14:52`.
- Runtime depends on hardware, model choice, and batch size.

### Known nondeterminism / caveats
- GPU execution, FAISS-based retrieval, and notebook-interactive execution can introduce run-to-run variation.
- Results can vary slightly across hardware and drivers even with fixed seeds and pinned software.

## Project Structure

```text
.
├── evaluation/
│   ├── evaluate-amharic-colbert-passage-retrieval.ipynb
│   ├── evaluate-amharic-embedding-passage-retrieval.ipynb
│   ├── evaluate-amharic-rerankers-passage-retrieval.ipynb
│   └── evaluate-amharic-splade-passage-retrieval.ipynb
├── preprocessing/
│   └── hard-negatives-mining-amharic-retrieval-dataset.ipynb
├── training/
│   ├── colbert-amharic/
│   │   ├── train-colbert-amharic-base.ipynb
│   │   └── train-colbert-amharic-medium.ipynb
│   ├── crossencoder-amharic/
│   │   ├── train-roberta-amharic-reranker-base.ipynb
│   │   └── train-roberta-amharic-reranker-medium.ipynb
│   ├── embeddings-amharic/
│   │   ├── train-roberta-amharic-embed-base.ipynb
│   │   └── train-roberta-amharic-embed-medium.ipynb
│   └── splade-amharic/
│       ├── train-splade-roberta-amharic-base.ipynb
│       └── train-splade-roberta-amharic-medium.ipynb
├── scripts/
│   ├── README.md                              # HPC setup instructions
│   ├── run_finetune_embeddinggemma.sbatch
│   ├── run_finetune_harrier.sbatch
│   ├── run_evaluate_gemma.sbatch
│   └── run_evaluate_harrier.sbatch
├── evaluate_ir.py                             # CLI evaluation script
├── finetune_embeddinggemma_amharic.py         # CLI fine-tuning script (EmbeddingGemma)
├── finetune_harrier_amharic.py                # CLI fine-tuning script (Harrier)
├── LICENSE
├── CITATION.cff
├── README.md
├── amharicir-environment.yml
└── requirements.txt
```

## License

This project is released under the MIT License. See `LICENSE`.

## Citation

GitHub citation metadata is available in `CITATION.cff`.

If you use this repository, please cite:

```bibtex
@inproceedings{alemneh2026amharicir,
  title     = {The Multilingual Curse at the Retrieval Layer: Evidence from Amharic},
  author    = {Alemneh, Yosef Worku and Mekonnen, Kidist Amde Mekand de Rijke, Maarten},
  booktitle = {Proceedings of the 1st Workshop on Multilinguality in the Era of Large Language Models (MeLLM), ACL 2026},
  year      = {2026},
}
```

## Troubleshooting / FAQ

Q: `ModuleNotFoundError` when opening notebooks.  
A: Activate the virtual environment and reinstall dependencies:
```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Q: Notebook tries to run on CUDA but no GPU is available.  
A: Use notebooks/cells that already support fallback (`torch.cuda.is_available()`), or set model/device cells to CPU explicitly where needed.

Q: Results differ from previous runs.  
A: Check seed usage (`seed=42` appears in several training notebooks), hardware differences (A100/T4/CPU), and CUDA/driver differences.

Q: Where are CLI scripts for end-to-end pipeline runs?  
A: See `evaluate_ir.py` (evaluation), `finetune_embeddinggemma_amharic.py`, and `finetune_harrier_amharic.py` (fine-tuning). SLURM job scripts that wrap these are in `scripts/`.
