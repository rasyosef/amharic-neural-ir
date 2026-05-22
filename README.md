

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
- Fine-tuned multilingual models: [kiyam/amharic-fine-tuned-multilingual-retrievers](https://huggingface.co/collections/kiyam/amharic-fine-tuned-multilingual-retrievers)

**Monolingual Amharic models**
- [rasyosef/embedding-amharic-base](https://huggingface.co/rasyosef/embedding-amharic-base)
- [rasyosef/embedding-amharic-medium](https://huggingface.co/rasyosef/embedding-amharic-medium)
- [rasyosef/colbert-amharic-base](https://huggingface.co/rasyosef/colbert-amharic-base)  
- [rasyosef/colbert-amharic-medium](https://huggingface.co/rasyosef/colbert-amharic-medium)  
- [rasyosef/splade-amharic-base](https://huggingface.co/rasyosef/splade-amharic-base)
- [rasyosef/splade-amharic-medium](https://huggingface.co/rasyosef/splade-amharic-medium)
- [rasyosef/reranker-amharic-base](https://huggingface.co/rasyosef/reranker-amharic-base)
- [rasyosef/reranker-amharic-medium](https://huggingface.co/rasyosef/reranker-amharic-medium)

**Amharic-fine-tuned multilingual models**
- [kiyam/EmbeddingGemma-300M-Amharic](https://huggingface.co/kiyam/EmbeddingGemma-300M-Amharic) — MRR@10: 0.718, NDCG@10: 0.753
- [kiyam/Harrier-270M-Amharic](https://huggingface.co/kiyam/Harrier-270M-Amharic) — MRR@10: 0.760, NDCG@10: 0.795

### Retriever Models Eval results
First-stage retrieval results on the Amharic Passage Retrieval Dataset V2.

| Model                                                    | Params (M) |          R@5 |         R@10 |       MRR@10 |      NDCG@10 |
| -------------------------------------------------------- | ---------: | -----------: | -----------: | -----------: | -----------: |
| ***Monolingual Amharic retrievers introduced in this work***|            |              |              |              |              |
| [`splade-amharic-medium`](https://huggingface.co/rasyosef/splade-amharic-medium)                                    |         42 |        0.858 |        0.896 |        0.728 |        0.769 |
| [`splade-amharic-base`](https://huggingface.co/rasyosef/splade-amharic-base)                                      |        110 |        0.871 |        0.906 |        0.754 |        0.792 |
| [`embedding-amharic-medium`](https://huggingface.co/rasyosef/embedding-amharic-medium)                                     |         42 |        0.843 |        0.888 |        0.744 |        0.779 |
| [`embedding-amharic-base`](https://huggingface.co/rasyosef/embedding-amharic-base)                                       |        110 |        0.870 |        0.907 |        0.774 |        0.807 |
| [`colbert-amharic-medium`](https://huggingface.co/rasyosef/colbert-amharic-medium)                                   |         42 |         0.882|        0.913 |        0.778 |        0.811 |
| [`colbert-amharic-base`](https://huggingface.co/rasyosef/colbert-amharic-base)                                     |        110 |   **0.902†** |   **0.930†** |   **0.803†** |   **0.835†** |
| ***Amharic-fine-tuned multilingual dense retrievers***   |            |              |              |              |              |
| [`EmbeddingGemma-300M-Amharic`](https://huggingface.co/kiyam/EmbeddingGemma-300M-Amharic)      |        300 |        0.813 |        0.862 |        0.718 |        0.753 |
| [`Harrier-270M-Amharic`](https://huggingface.co/kiyam/Harrier-270M-Amharic)                                 |        270 |        0.860 |        0.903 |        0.760 |        0.795 |
| ***Monolingual Amharic retrievers from prior work***     |            |              |              |              |              |
| `roberta-amharic-text-embedding-medium`                    |         42 |        0.750 |        0.807 |        0.616 |        0.662 |
| `roberta-amharic-text-embedding-base`                      |        110 |        0.790 |        0.844 |        0.657 |        0.703 |
| `colbert-roberta-amharic-base`                             |        110 |        0.860 |        0.899 |        0.736 |        0.776 |
| ***Zero-shot multilingual dense retrievers***            |            |              |              |              |              |
| `embeddinggemma-300m`                                      |        300 |        0.558 |        0.621 |        0.448 |        0.489 |
| `gte-multilingual-base`                                    |        305 |        0.690 |        0.755 |        0.557 |        0.605 |
| `harrier-oss-v1-270m`                                      |        270 |        0.697 |        0.753 |        0.576 |        0.619 |
| `multilingual-e5-large-instruct`                           |        560 |        0.736 |        0.791 |        0.603 |        0.648 |
| `snowflake-arctic-embed-l-v2.0`                            |        568 |        0.795 |        0.848 |        0.653 |        0.701 |
| ***Sparse lexical retrieval***                           |            |              |              |              |              |
| `BM25`                                                     |         -- |        0.734 |        0.789 |        0.612 |        0.655 |

### Reranker Models eval results
Two-stage re-ranking results on the Amharic Passage Retrieval Dataset V2.

| Model                      | Params (M) |    MRR@10 |   NDCG@10 |
| -------------------------- | ---------: | --------: | --------: |
| `embedding-amharic-base`     |         110 |     0.774 |     0.807 |
| + [`reranker-amharic-medium`](https://huggingface.co/rasyosef/reranker-amharic-medium) |         42 |     0.805 |     0.835 |
| **+ [`reranker-amharic-base`](https://huggingface.co/rasyosef/reranker-amharic-base)** |         110 | **0.830** | **0.856** |


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
