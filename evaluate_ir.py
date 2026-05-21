# -*- coding: utf-8 -*-

import argparse

import torch
from datasets import load_dataset, concatenate_datasets

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.util import cos_sim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace model ID or local path to evaluate",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="yosefw/amharic-news-retrieval-dataset-v2-with-negatives-V2",
    )
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument(
        "--matryoshka_dims",
        type=int,
        nargs="+",
        default=None,
        help="Matryoshka dims to evaluate. Defaults to [native_dim, 256].",
    )
    parser.add_argument(
        "--query_prompt",
        type=str,
        default=None,
        help="Optional prompt prepended to queries at inference time.",
    )
    return parser.parse_args()


def build_ir_data(dataset_name):
    dataset = load_dataset(dataset_name)
    dataset = dataset.rename_column("query", "anchor")
    dataset = dataset.rename_column("passage", "positive")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    corpus_dataset = concatenate_datasets([train_dataset, test_dataset])

    corpus = dict(zip(corpus_dataset["passage_id"], corpus_dataset["positive"]))
    queries = dict(zip(test_dataset["query_id"], test_dataset["anchor"]))

    relevant_docs = {}
    for row in test_dataset:
        relevant_docs[row["query_id"]] = [row["passage_id"]]

    return corpus, queries, relevant_docs


def build_evaluator(model, queries, corpus, relevant_docs, eval_batch_size, matryoshka_dims, query_prompt=None):
    embed_dim = model.get_sentence_embedding_dimension()

    if matryoshka_dims is None:
        matryoshka_dims = [embed_dim, 256]

    evaluators = []
    for dim in matryoshka_dims:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,
            score_functions={"cosine": cos_sim},
            batch_size=eval_batch_size,
            corpus_chunk_size=2048,
            show_progress_bar=True,
            mrr_at_k=[10],
            ndcg_at_k=[10],
            accuracy_at_k=[5, 10, 50, 100],
            precision_recall_at_k=[5, 10, 50, 100],
            query_prompt=query_prompt,
        )
        evaluators.append(ir_evaluator)

    return SequentialEvaluator(evaluators), matryoshka_dims


def print_eval_results(results, matryoshka_dims):
    metrics = ["mrr@10", "ndcg@10"] + [f"recall@{k}" for k in [5, 10, 50, 100]]
    for dim in matryoshka_dims:
        print(f"\n--- dim {dim} ---")
        for metric in metrics:
            key = f"dim_{dim}_cosine_{metric}"
            if key in results:
                print(f"  {metric}: {results[key]:.4f}")


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print(f"Model:  {args.model_id}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print("=" * 80)

    print("Loading dataset...")
    corpus, queries, relevant_docs = build_ir_data(args.dataset_name)
    print(f"Corpus: {len(corpus)} passages | Queries: {len(queries)}")

    print("Loading model...")
    model = SentenceTransformer(args.model_id, device=device)

    if model.max_seq_length > args.max_seq_length:
        model.max_seq_length = args.max_seq_length
    print(f"Max seq length: {model.max_seq_length}")

    if args.query_prompt:
        print(f"Query prompt: {args.query_prompt!r}")

    evaluator, matryoshka_dims = build_evaluator(
        model=model,
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        eval_batch_size=args.eval_batch_size,
        matryoshka_dims=args.matryoshka_dims,
        query_prompt=args.query_prompt,
    )

    print("\nRunning evaluation...")
    results = evaluator(model)
    print("\n=== Results ===")
    print_eval_results(results, matryoshka_dims)
    print("\nDone.")


if __name__ == "__main__":
    main()
