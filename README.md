# The Multilingual Curse at the Retrieval Layer: Evidence from Amharic

[![ACL 2026](https://img.shields.io/badge/ACL-2026-blue)](https://2026.aclweb.org/)
[![MeLLM Workshop](https://img.shields.io/badge/Workshop-MeLLM-informational)](https://mellm.org/)
[![Hugging Face models](https://img.shields.io/badge/%F0%9F%A4%97-Amharic%20models-yellow)](https://huggingface.co/collections/rasyosef/amharic-neural-ir-models)
[![Hugging Face multilingual models](https://img.shields.io/badge/%F0%9F%A4%97-Multilingual%20models-yellow)](https://huggingface.co/collections/kiyam/amharic-fine-tuned-multilingual-retrievers)
[![License](https://img.shields.io/badge/license-LICENSE-lightgrey)](LICENSE)

This repository accompanies the ACL 2026 MeLLM Workshop paper **"The Multilingual Curse at the Retrieval Layer: Evidence from Amharic."** It provides notebook and CLI workflows for dense retrieval, late interaction (ColBERT-style), sparse retrieval (SPLADE-style), and cross-encoder reranking in Amharic.

**Paper**: https://arxiv.org/abs/2605.24556

**Core artifacts**
- **Benchmark**: Amharic Passage Retrieval Dataset V2 with a fixed 90/10 train–test split (68,000 query–passage pairs).
- **Model suite**: Amharic-specific checkpoints spanning `dense bi-encoders`, `late-interaction (ColBERT-style)`, `learned sparse retrievers (SPLADE-style)`, and `cross-encoder rerankers`.
- **Workflows**: notebook implementations for `preprocessing`, `training`, `evaluation`, `indexing`, `search`, and `RAG` plus CLI/SLURM scripts for selected fine-tuning and evaluation runs.

**Hugging Face resources**
- **Dataset:** [rasyosef/Amharic-Passage-Retrieval-Dataset-V2](https://huggingface.co/datasets/rasyosef/Amharic-Passage-Retrieval-Dataset-V2)
- **Monolingual Amharic models:** [rasyosef/amharic-neural-ir-models](https://huggingface.co/collections/rasyosef/amharic-neural-ir-models)
- **Fine-tuned multilingual models:** [kiyam/amharic-fine-tuned-multilingual-retrievers](https://huggingface.co/collections/kiyam/amharic-fine-tuned-multilingual-retrievers)

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

**Indexing and Search notebooks**

To see the models in action, check out the following `notebooks`. 
- [Amharic Embedding, Reranking & RAG with LlamaIndex](https://colab.research.google.com/github/rasyosef/amharic-neural-ir/blob/main/indexing-and-search/Amharic%20Embedding,%20Reranking%20&%20RAG%20with%20LlamaIndex.ipynb) 

> This hands-on guide demonstrates how to use our custom embedding models and cross-encoder rerankers alongside **LlamaIndex** to implement a robust **two-stage retrieval** pipeline and a complete **RAG** (Retrieval-Augmented Generation) system.

- [Sparse Retrieval with Amharic SPLADE and splade-index](https://colab.research.google.com/github/rasyosef/amharic-neural-ir/blob/main/indexing-and-search/Sparse%20Retrieval%20with%20Amharic%20SPLADE%20and%20splade-index.ipynb)
- [Late-Interaction Retrieval with Amharic ColBERT and PLAID](https://colab.research.google.com/github/rasyosef/amharic-neural-ir/blob/main/indexing-and-search/Retrieval%20with%20Amharic%20ColBERT%20and%20PLAID.ipynb)

> [!TIP]
> Open any notebook directly in **Google Colab** by replacing `github.com` in the notebook URL with `colab.research.google.com/github/`.
>
> **Example**
>
> Original:
> `https://github.com/rasyosef/amharic-neural-ir/blob/main/training/embeddings-amharic/train-roberta-amharic-embed-base.ipynb`
>
> Colab:
> `https://colab.research.google.com/github/rasyosef/amharic-neural-ir/blob/main/training/embeddings-amharic/train-roberta-amharic-embed-base.ipynb`

## Project Structure

```text
amharic-neural-ir/
│
├── preprocessing/
│   └── hard-negatives-mining-amharic-retrieval-dataset.ipynb
│
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
│
├── evaluation/
│   ├── evaluate-amharic-colbert-passage-retrieval.ipynb
│   ├── evaluate-amharic-embedding-passage-retrieval.ipynb
│   ├── evaluate-amharic-rerankers-passage-retrieval.ipynb
│   └── evaluate-amharic-splade-passage-retrieval.ipynb
│
├── indexing-and-search/
│   ├── Amharic Embedding, Reranking & RAG with LlamaIndex.ipynb
│   ├── Sparse Retrieval with Amharic SPLADE and splade-index.ipynb
│   └── Retrieval with Amharic ColBERT and PLAID.ipynb
│
├── scripts/
│   ├── README.md
│   ├── run_finetune_embeddinggemma.sbatch
│   ├── run_finetune_harrier.sbatch
│   ├── run_evaluate_gemma.sbatch
│   └── run_evaluate_harrier.sbatch
│
├── evaluate_ir.py                        # CLI: evaluate retrieval models
├── finetune_embeddinggemma_amharic.py    # CLI: fine-tune EmbeddingGemma
├── finetune_harrier_amharic.py           # CLI: fine-tune Harrier
├── amharicir-environment.yml             # Conda environment (Python 3.10)
├── requirements.txt                      # pip dependencies
├── CITATION.cff                          # Citation metadata
├── LICENSE                               # MIT License
└── README.md
```


#### Retriever Model Evaluation Results
First-stage retrieval results on the Amharic Passage Retrieval Dataset V2.

| Model                                                    | Params (M) |          R@5 |         R@10 |       MRR@10 |      NDCG@10 |
| -------------------------------------------------------- | ---------: | -----------: | -----------: | -----------: | -----------: |
| *Monolingual Amharic retrievers introduced in this work* |            |              |              |              |              |
| [`splade-amharic-medium`](https://huggingface.co/rasyosef/splade-amharic-medium)            |         42 |        85.8 |        89.6 |        72.8 |        76.9 |
| [`splade-amharic-base`](https://huggingface.co/rasyosef/splade-amharic-base)                |        110 |        87.1 |        90.6 |        75.4 |        79.2 |
| [`embedding-amharic-medium`](https://huggingface.co/rasyosef/embedding-amharic-medium)      |         42 |        84.3 |        88.8 |        74.4 |        77.9 |
| [`embedding-amharic-base`](https://huggingface.co/rasyosef/embedding-amharic-base)          |        110 |        87.0 |        90.7 |        77.4 |        80.7 |
| [`colbert-amharic-medium`](https://huggingface.co/rasyosef/colbert-amharic-medium)          |         42 |        88.2 |        91.3 |        77.8 |        81.1 |
| **[`colbert-amharic-base†`](https://huggingface.co/rasyosef/colbert-amharic-base)**              |        110 |   **90.2†** |   **93.0†** |   **80.3†** |   **83.5†** |
| *Amharic-fine-tuned multilingual dense retrievers*       |            |              |              |              |              |
| [`EmbeddingGemma-300M-Amharic`](https://huggingface.co/kiyam/EmbeddingGemma-300M-Amharic)   |        300 |        81.3 |        86.2 |        71.8 |        75.3 |
| [`Harrier-270M-Amharic`](https://huggingface.co/kiyam/Harrier-270M-Amharic)                 |        270 |        86.0 |        90.3 |        76.0 |        79.5 |
| *Monolingual Amharic retrievers from prior work*           |            |             |             |             |             |
| `roberta-amharic-text-embedding-medium`                    |         42 |        75.0 |        80.7 |        61.6 |        66.2 |
| `roberta-amharic-text-embedding-base`                      |        110 |        79.0 |        84.4 |        65.7 |        70.3 |
| `colbert-roberta-amharic-base`                             |        110 |        86.0 |        89.9 |        73.6 |        77.6 |
| *Zero-shot multilingual dense retrievers*                  |            |             |             |              |              |
| `embeddinggemma-300m`                                      |        300 |        55.8 |        62.1 |        44.8 |        48.9 |
| `gte-multilingual-base`                                    |        305 |        69.0 |        75.5 |        55.7 |        60.5 |
| `harrier-oss-v1-270m`                                      |        270 |        69.7 |        75.3 |        57.6 |        61.9 |
| `multilingual-e5-large-instruct`                           |        560 |        73.6 |        79.1 |        60.3 |        64.8 |
| `snowflake-arctic-embed-l-v2.0`                            |        568 |        79.5 |        84.8 |        65.3 |        70.1 |
| *Sparse lexical retrieval*                                 |            |             |             |              |              |
| `BM25`                                                     |         -- |        73.4 |        78.9 |         61.2 |         65.5 |

> **†** Best overall performance across evaluated retriever models.

#### Reranker Model Evaluation Results
Two-stage re-ranking results on the Amharic Passage Retrieval Dataset V2.

| Model                      | Params (M) |    MRR@10 |   NDCG@10 |
| -------------------------- | ---------: | --------: | --------: |
| `embedding-amharic-base`     |         110 |     77.4 |     80.7 |
| + [`reranker-amharic-medium`](https://huggingface.co/rasyosef/reranker-amharic-medium) |         42 |     80.5 |     83.5 |
| **+ [`reranker-amharic-base†`](https://huggingface.co/rasyosef/reranker-amharic-base)** |         110 | **83.0†** | **85.6†** |

> **†** Best overall performance across evaluated reranker models.

### Using the models

#### Direct Usage (Sentence Transformers)
First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load the models and run inference.

#### 1. Embedding Models
Embedding models convert text into dense vector representations that can be used for:

- Semantic search
- Information retrieval
- Clustering
- Similarity search

##### Monolingual
- Model: `rasyosef/embedding-amharic-base`
- Optimized for Amharic semantic search and retrieval tasks.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("rasyosef/embedding-amharic-base")

# What is the capital of Ethiopia? / France
queries = ['የኢትዮጵያ ዋና ከተማ ማናት?', 'የፈረንሳይ ዋና ከተማ ማናት?'] 

# Addis Ababa, Gondar, Paris, London, Washington D.C.
documents = ['አዲስ አበባ', 'ጎንደር', 'ፓሪስ', 'ለንደን', 'ዋሽንግተን ዲሲ'] 

# Compute embeddings
query_embeddings = model.encode_query(queries) # [2, 768]
document_embeddings = model.encode_document(documents) # [5, 768]

# Calculate semantic similarity
similarities = model.similarity(
    query_embeddings, 
    document_embeddings
)

print(similarities)
# tensor([[0.5075, 0.3114, 0.0798, 0.1967, 0.1340],
#         [0.1777, 0.0770, 0.5714, 0.2596, 0.1076]])
```

##### Fine-tuned Multilingual
- Model: `kiyam/Harrier-270M-Amharic`
- Multilingual embedding model further fine-tuned for Amharic retrieval tasks.

```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("kiyam/Harrier-270M-Amharic")
# Run inference
sentences = [
    'ለውጭ ገበያ በሚቀርበው የኢትዮጵያ ቡና ላይ የተጋረጠው ፈተና',
    'የኢትዮጵያ ዋነኛ የውጭ ምንዛሬ ምንጭ የሆነው ወደ ውጭ የሚላክ ቡና ዘርፍ በአሁኑ ጊዜ ከፍተኛ ውጥረት ውስጥ ገብቷል።',
    'የቻይናው ፕሬዝዳንት ዚ ጂንፒንግ ከትራምፕ ጋር ባደረጉት ጉባኤ ትኩረታቸው በሁለቱ ሀገራት መካከል ለወራት ከተፈጠረ ውጥረት እና የንግድ ጦርነት በኋላ የተረገጋጋ ግንኙነትን ማስቀጠል ነበር።',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

#### 2. Rerankers/Cross-Encoders
Cross-encoders jointly encode a query-document pair and produce a relevance score.
They are commonly used to rerank candidates retrieved by a first-stage retriever.

- Model: `rasyosef/reranker-amharic-base`

```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("rasyosef/reranker-amharic-base")

# Get scores for pairs of texts
pairs = [
    ['ለውጭ ገበያ በሚቀርበው የኢትዮጵያ ቡና ላይ የተጋረጠው ፈተና', 'የኢትዮጵያ ዋነኛ የውጭ ምንዛሬ ምንጭ የሆነው ወደ ውጭ የሚላክ ቡና ዘርፍ በአሁኑ ጊዜ ከፍተኛ ውጥረት ውስጥ ገብቷል።'],
    ['ለውጭ ገበያ በሚቀርበው የኢትዮጵያ ቡና ላይ የተጋረጠው ፈተና', 'የቻይናው ፕሬዝዳንት ዚ ጂንፒንግ ከትራምፕ ጋር ባደረጉት ጉባኤ ትኩረታቸው በሁለቱ ሀገራት መካከል ለወራት ከተፈጠረ ውጥረት እና የንግድ ጦርነት በኋላ የተረገጋጋ ግንኙነትን ማስቀጠል ነበር።']
]
scores = model.predict(pairs)
print(scores.shape)
# (2,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'ለውጭ ገበያ በሚቀርበው የኢትዮጵያ ቡና ላይ የተጋረጠው ፈተና',
    [
        'የኢትዮጵያ ዋነኛ የውጭ ምንዛሬ ምንጭ የሆነው ወደ ውጭ የሚላክ ቡና ዘርፍ በአሁኑ ጊዜ ከፍተኛ ውጥረት ውስጥ ገብቷል።',
        'የቻይናው ፕሬዝዳንት ዚ ጂንፒንግ ከትራምፕ ጋር ባደረጉት ጉባኤ ትኩረታቸው በሁለቱ ሀገራት መካከል ለወራት ከተፈጠረ ውጥረት እና የንግድ ጦርነት በኋላ የተረገጋጋ ግንኙነትን ማስቀጠል ነበር።',
    ]
)
print(ranks)
# [{'corpus_id': 0, 'score': np.float32(0.9555243)}, {'corpus_id': 1, 'score': np.float32(0.0012893651)}]
```

#### 3. SPLADE / Sparse Encoders
SPLADE models generate sparse lexical-semantic representations compatible with inverted indexes.

- Model: `rasyosef/splade-amharic-base`

```python
from sentence_transformers import SparseEncoder

# Download from the 🤗 Hub
model = SparseEncoder("rasyosef/splade-amharic-base")
# Run inference
sentences = [
    'ለውጭ ገበያ በሚቀርበው የኢትዮጵያ ቡና ላይ የተጋረጠው ፈተና',
    'የኢትዮጵያ ዋነኛ የውጭ ምንዛሬ ምንጭ የሆነው ወደ ውጭ የሚላክ ቡና ዘርፍ በአሁኑ ጊዜ ከፍተኛ ውጥረት ውስጥ ገብቷል።',
    'የቻይናው ፕሬዝዳንት ዚ ጂንፒንግ ከትራምፕ ጋር ባደረጉት ጉባኤ ትኩረታቸው በሁለቱ ሀገራት መካከል ለወራት ከተፈጠረ ውጥረት እና የንግድ ጦርነት በኋላ የተረገጋጋ ግንኙነትን ማስቀጠል ነበር።',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 32000]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[45.2024, 11.8912,  0.0000],
#         [11.8912, 29.1348,  6.6066],
#         [ 0.0000,  6.6066, 39.7972]])

```

#### 4. ColBERT / Late-Interaction
ColBERT models use late interaction between token embeddings for high-quality retrieval.

- Model: `rasyosef/colbert-amharic-base`

First install the PyLate library:

```bash
pip install -U pylate
```

```python
from pylate import models

# Download from the 🤗 Hub
model = models.ColBERT(
    model_name_or_path="rasyosef/colbert-amharic-base",
)
# Run inference
sentences = [
    'ለውጭ ገበያ በሚቀርበው የኢትዮጵያ ቡና ላይ የተጋረጠው ፈተና',
    'የኢትዮጵያ ዋነኛ የውጭ ምንዛሬ ምንጭ የሆነው ወደ ውጭ የሚላክ ቡና ዘርፍ በአሁኑ ጊዜ ከፍተኛ ውጥረት ውስጥ ገብቷል።',
    'የቻይናው ፕሬዝዳንት ዚ ጂንፒንግ ከትራምፕ ጋር ባደረጉት ጉባኤ ትኩረታቸው በሁለቱ ሀገራት መካከል ለወራት ከተፈጠረ ውጥረት እና የንግድ ጦርነት በኋላ የተረገጋጋ ግንኙነትን ማስቀጠል ነበር።',
]
embeddings = model.encode(
    sentences,
    is_query=True,
)
print(embeddings[0].shape)
# (32, 128)

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[32.0000, 20.4463,  2.9574],
#         [19.6756, 32.0000, 11.4214],
#         [ 2.8087, 11.9757, 32.0000]])
```

## Notebook-first workflow

This codebase is organized primarily as Jupyter notebooks, with standalone Python scripts for selected fine-tuning and evaluation runs. The goal is to keep the full pipeline easy to follow and modify step-by-step, especially for practitioners. Because the dataset used in these workflows is relatively small, we keep the main experiments and analysis in notebook format for clarity and quick iteration.

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


## License

This project is released under the MIT License. See `LICENSE`.

## Citation

GitHub citation metadata is available in `CITATION.cff`.

If you use this repository, please cite:

```bibtex
@inproceedings{alemneh2026amharicir,
  title     = {The Multilingual Curse at the Retrieval Layer: Evidence from Amharic},
  author    = {Alemneh, Yosef Worku and Mekonnen, Kidist Amde and de Rijke, Maarten},
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
