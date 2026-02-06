
# AmharicIR: A Unified Resource for Amharic Neural Retrieval

[![SIGIR 2026](https://img.shields.io/badge/SIGIR-2026-blue)](#)
[![Submission](https://img.shields.io/badge/submission-83-informational)](#)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/collections/rasyosef/amharic-neural-ir-models)
[![License](https://img.shields.io/badge/license-LICENSE-lightgrey)](LICENSE)

This repository accompanies the SIGIR 2026 paper **“AmharicIR: A Unified Resource for Amharic Neural Retrieval Models and Benchmarks.”** It provides evaluation notebooks today and is organized to grow into a fully reproducible pipeline for training, indexing, retrieval, reranking, and benchmarking across multiple neural IR families in Amharic.

**Core artifacts from the paper**
- **Benchmark**: Amharic Passage Retrieval Dataset V2 with a fixed 90/10 train–test split (68,000 query–passage pairs).
- **Model suite**: Amharic-specific checkpoints spanning dense bi-encoders, late-interaction (ColBERT-style), learned sparse retrievers (SPLADE-style), and cross-encoder rerankers.
- **Reproducible pipelines**: standardized preprocessing, indexing, retrieval, reranking, and evaluation settings with pinned configs and metadata logging.

**Hugging Face resources**
- Dataset: [rasyosef/Amharic-Passage-Retrieval-Dataset-V2](https://huggingface.co/datasets/rasyosef/Amharic-Passage-Retrieval-Dataset-V2)
- Model collection: [rasyosef/amharic-neural-ir-models](https://huggingface.co/collections/rasyosef/amharic-neural-ir-models)

**Models used in current notebooks (examples)**
- [rasyosef/RoBERTa-Amharic-Embed-Base](https://huggingface.co/rasyosef/RoBERTa-Amharic-Embed-Base)
- [rasyosef/RoBERTa-Amharic-Embed-Medium](https://huggingface.co/rasyosef/RoBERTa-Amharic-Embed-Medium)
- [rasyosef/colbert-roberta-amharic-base](https://huggingface.co/rasyosef/colbert-roberta-amharic-base)
- [rasyosef/SPLADE-RoBERTa-Amharic-Base](https://huggingface.co/rasyosef/SPLADE-RoBERTa-Amharic-Base)
- [rasyosef/SPLADE-RoBERTa-Amharic-Medium](https://huggingface.co/rasyosef/SPLADE-RoBERTa-Amharic-Medium)
- [rasyosef/RoBERTa-Amharic-Reranker-Base](https://huggingface.co/rasyosef/RoBERTa-Amharic-Reranker-Base)

## Setup

**Environment**
```bash
conda activate pag-env
```

**Install dependencies**
The evaluation notebooks use the following packages (exact versions will be pinned in a future `environment.yml` or `requirements.txt`):
```bash
pip install -U sentence-transformers datasets transformers
pip install -U faiss-cpu  # or faiss-gpu if available
pip install -U pylate beir ranx
pip install -U torch
```

Optional: `wandb` is used in the notebooks but is run in disabled mode by default.

## Repository Structure

Current
- `evaluation/` notebook-based evaluation for each model family.
- `Amharic_Retrieval_II__SIGIR2026.pdf` paper.
- `LICENSE` MIT license.

Planned (roadmap; scripts to be added)
- `configs/` pinned configs for preprocessing, indexing, retrieval, reranking, and evaluation.
- `data/` standardized dataset layout and cached artifacts.
- `models/` local checkpoints and adapters for reproducible training.
- `scripts/` CLI entry points for end-to-end workflows.
- `indexes/` built search indexes (FAISS, HNSW, inverted indices).
- `runs/` experiment outputs, metrics, and metadata.
- `outputs/` ranked runs and evaluation reports.

## Path Conventions and Expected Outputs

This project will standardize paths to make runs fully reproducible and auditable. The following conventions reflect the planned pipeline and are used in the README so future scripts and notebooks remain consistent.

- `data/raw/` raw dataset downloads.
- `data/processed/` normalized and deduplicated corpus, queries, and qrels.
- `configs/*.yaml` immutable config snapshots for each experiment.
- `indexes/{model_family}/{model_name}/` index artifacts for retrieval.
- `runs/{date}/{experiment_name}/` metrics, logs, and metadata.
- `outputs/{experiment_name}/` ranked results and evaluation summaries.

## Evaluation Workflows (Current)

All evaluations are based on the Amharic Passage Retrieval Dataset V2 and follow the paper’s protocol: retrieve **top-100** candidates per query for first-stage models and report **MRR@10** and **NDCG@10**. Cross-encoders rerank the top-100 candidates and are evaluated at cutoff 10.

### 1) Dense Bi-Encoder Evaluation
Notebook: `evaluation/evaluate-amharic-embedding-passage-retrieval.ipynb`

What it does
- Loads dataset from Hugging Face.
- Builds a corpus of passages and evaluates a Sentence-Transformers encoder.
- Runs IR metrics via `InformationRetrievalEvaluator`.

### 2) Late-Interaction (ColBERT-Style) Evaluation
Notebook: `evaluation/evaluate-amharic-colbert-passage-retrieval.ipynb`

What it does
- Uses PyLate and a Voyager HNSW index.
- Evaluates ColBERT-style models with multi-vector representations.

### 3) Learned Sparse (SPLADE-Style) Evaluation
Notebook: `evaluation/evaluate-amharic-splade-passage-retrieval.ipynb`

What it does
- Uses Sentence-Transformers `SparseEncoder` for SPLADE-style retrieval.
- Evaluates inverted-index sparse vectors and metrics with `SparseInformationRetrievalEvaluator`.

### 4) Two-Stage Reranking
Notebook: `evaluation/evaluate-amharic-rerankers-passage-retrieval.ipynb`

What it does
- Builds a first-stage FAISS index over dense embeddings.
- Reranks top-100 candidates with a Cross-Encoder.
- Evaluates MRR@10 and NDCG@10 for reranked results.

## Reproducibility and Observability

The paper emphasizes stable evaluation through pinned configs and logged metadata. The repository will enforce these practices in upcoming scripts:
- Fixed random seed (`42`) and deterministic settings where possible.
- Logged metadata per run: software versions, commit hash, config file, retrieval depth, and indexing parameters.
- Immutable config snapshots stored under `configs/` and `runs/`.
- Versioned dataset and checkpoint releases via Hugging Face.

## Roadmap: How Components Interact

The planned pipeline is organized as a sequence of modular, reusable steps. Each step will have a dedicated CLI script and a config file, so new experiments are composable and fully traceable.

1. **Preprocessing** — Input: Hugging Face dataset. Output: `data/processed/` (normalized corpus, queries, qrels).
2. **Training** — Input: processed dataset. Output: `models/` checkpoints and `runs/` metrics. Notes: training recipes follow the paper’s hyperparameters by model family.
3. **Indexing** — Input: model checkpoint + processed corpus. Output: `indexes/` (FAISS, HNSW/Voyager, or inverted index).
4. **First-Stage Retrieval** — Input: index + queries. Output: `outputs/` ranked runs and `runs/` metrics.
5. **Reranking** — Input: top-100 candidates + cross-encoder. Output: `outputs/` reranked runs and `runs/` metrics.
6. **Evaluation** — Input: ranked runs + qrels. Output: `outputs/` reports (MRR@10, NDCG@10, Recall@k).

## Extending the Repository

Planned scripts will make extensions straightforward and consistent:
- `scripts/preprocess.py` will normalize and deduplicate the dataset with fixed rules.
- `scripts/train_dense.py`, `scripts/train_colbert.py`, `scripts/train_splade.py`, `scripts/train_reranker.py` will mirror the paper’s training schedules and tokenization choices.
- `scripts/build_index.py` will build FAISS, Voyager HNSW, or inverted indexes.
- `scripts/retrieve.py` and `scripts/rerank.py` will produce standardized runs and metadata.
- `scripts/evaluate.py` will compute metrics and export reports.

## Citation

If you use this repository or the released benchmarks, please cite the SIGIR’26 paper.

## License

MIT License. See `LICENSE`.
