

# AmharicIR: A Unified Resource for Amharic Neural Retrieval

[![SIGIR 2026](https://img.shields.io/badge/SIGIR-2026-blue)](#)
[![Submission](https://img.shields.io/badge/submission-83-informational)](#)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/collections/rasyosef/amharic-neural-ir-models)
[![License](https://img.shields.io/badge/license-LICENSE-lightgrey)](LICENSE)

This repository accompanies the SIGIR 2026 paper **“AmharicIR: A Unified Resource for Amharic Neural Retrieval Models and Benchmarks.”** It provides notebook-based training and evaluation workflows for dense retrieval, late interaction (ColBERT-style), sparse retrieval (SPLADE-style), and cross-encoder reranking in Amharic.

**Core artifacts**
- **Benchmark**: Amharic Passage Retrieval Dataset V2 with a fixed 90/10 train–test split (68,000 query–passage pairs).
- **Model suite**: Amharic-specific checkpoints spanning dense bi-encoders, late-interaction (ColBERT-style), learned sparse retrievers (SPLADE-style), and cross-encoder rerankers.
- **Workflows**: notebook implementations for preprocessing, training, and evaluation.

**Hugging Face resources**
- Dataset: [rasyosef/Amharic-Passage-Retrieval-Dataset-V2](https://huggingface.co/datasets/rasyosef/Amharic-Passage-Retrieval-Dataset-V2)
- Model collection: [rasyosef/amharic-neural-ir-models](https://huggingface.co/collections/rasyosef/amharic-neural-ir-models)

**Models used in current notebooks (examples)**
- [rasyosef/RoBERTa-Amharic-Embed-Base](https://huggingface.co/rasyosef/RoBERTa-Amharic-Embed-Base)
- [rasyosef/RoBERTa-Amharic-Embed-Medium](https://huggingface.co/rasyosef/RoBERTa-Amharic-Embed-Medium)
- [rasyosef/ColBERT-Amharic-Base](https://huggingface.co/rasyosef/ColBERT-Amharic-Base)
- [rasyosef/ColBERT-Amharic-Medium](https://huggingface.co/rasyosef/ColBERT-Amharic-Medium)
- [rasyosef/SPLADE-RoBERTa-Amharic-Base](https://huggingface.co/rasyosef/SPLADE-RoBERTa-Amharic-Base)
- [rasyosef/SPLADE-RoBERTa-Amharic-Medium](https://huggingface.co/rasyosef/SPLADE-RoBERTa-Amharic-Medium)
- [rasyosef/RoBERTa-Amharic-Reranker-Base](https://huggingface.co/rasyosef/RoBERTa-Amharic-Reranker-Base)
- [rasyosef/RoBERTa-Amharic-Reranker-Medium](https://huggingface.co/rasyosef/RoBERTa-Amharic-Reranker-Medium)


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
@misc{alemneh2026amharicir,
  title        = {AmharicIR: A Unified Resource for Amharic Neural Retrieval Models and Benchmarks},
  author       = {Alemneh, Yosef Worku and Mekonnen, Kidist Amde and de Rijke, Maarten},
  year         = {2026},
  note         = {Manuscript under review},
  howpublished = {GitHub repository},
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

Q: Where are CLI scripts/config files for end-to-end pipeline runs?  
A: They are not present in the current repository layout.
