# -*- coding: utf-8 -*-

import os
import argparse
import random

import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm.auto import tqdm

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import cos_sim


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="yosefw/amharic-news-retrieval-dataset-v2-with-negatives-V2",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/harrier-oss-v1-270m",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="harrier-270m-am-finetuned",
    )
    parser.add_argument("--num_train_epochs", type=int, default=6)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()


def build_relevance_dataset(dataset, seed=42):
    ds_rows = []

    for row in tqdm(dataset["train"], desc="Building training triples"):
        neg_passages = row["negative_passages"]

        # Take first two and last two negatives.
        # Remove duplicates safely in case there are fewer than 4 negatives.
        selected_negatives = neg_passages[:2] + neg_passages[-2:]

        seen = set()
        unique_negatives = []
        for neg in selected_negatives:
            passage = neg["passage"]
            if passage not in seen:
                unique_negatives.append(neg)
                seen.add(passage)

        for neg_passage in unique_negatives:
            ds_rows.append(
                {
                    "query_id": row["query_id"],
                    "passage_id": row["passage_id"],
                    "anchor": row["anchor"],
                    "positive": row["positive"],
                    "negative": neg_passage["passage"],
                }
            )

    return Dataset.from_list(ds_rows).shuffle(seed=seed)


def build_ir_data(dataset):
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    corpus_dataset = concatenate_datasets([train_dataset, test_dataset])

    corpus = dict(zip(corpus_dataset["passage_id"], corpus_dataset["positive"]))
    queries = dict(zip(test_dataset["query_id"], test_dataset["anchor"]))

    relevant_docs = {}
    for row in test_dataset:
        relevant_docs[row["query_id"]] = [row["passage_id"]]

    return corpus, queries, relevant_docs


def build_evaluator(model, queries, corpus, relevant_docs, eval_batch_size):
    embed_dim = model.get_sentence_embedding_dimension()
    matryoshka_dimensions = [embed_dim, 256]

    evaluators = []

    for dim in matryoshka_dimensions:
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
        )
        evaluators.append(ir_evaluator)

    return SequentialEvaluator(evaluators), matryoshka_dimensions


def print_eval_results(results, matryoshka_dimensions):
    metrics = ["mrr@10", "ndcg@10"] + [f"recall@{k}" for k in [5, 10, 50, 100]]
    for dim in matryoshka_dimensions:
        print(f"\n--- dim {dim} ---")
        for metric in metrics:
            key = f"dim_{dim}_cosine_{metric}"
            if key in results:
                print(f"  {metric}: {results[key]:.4f}")


def main():
    args = parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print("=" * 80)

    print("Loading dataset...")
    dataset = load_dataset(args.dataset_name)

    print("Renaming columns...")
    dataset = dataset.rename_column("query", "anchor")
    dataset = dataset.rename_column("passage", "positive")

    print("Preparing training triples...")
    relevance_dataset = build_relevance_dataset(dataset, seed=args.seed)

    print(relevance_dataset)

    print("Preparing IR evaluator data...")
    corpus, queries, relevant_docs = build_ir_data(dataset)

    print("Loading model...")
    model = SentenceTransformer(
        args.model_id,
        device=device,
        model_kwargs={"attn_implementation": "sdpa"},
        model_card_data=SentenceTransformerModelCardData(
            language="am",
            license="mit",
            model_name="Harrier OSS 270M Amharic",
        ),
    )

    if model.max_seq_length > args.max_seq_length:
        model.max_seq_length = args.max_seq_length

    print(f"Model max sequence length: {model.max_seq_length}")

    evaluator, matryoshka_dimensions = build_evaluator(
        model=model,
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        eval_batch_size=args.eval_batch_size,
    )

    print("Evaluating pretrained model...")
    baseline_results = evaluator(model)
    print_eval_results(baseline_results, matryoshka_dimensions)

    print("Creating loss...")
    inner_train_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(
        model,
        inner_train_loss,
        matryoshka_dims=matryoshka_dimensions,
    )

    print("Creating training arguments...")
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_ratio=0.025,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        bf16=torch.cuda.is_available(),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        disable_tqdm=False,
        report_to=args.report_to if args.report_to != "none" else [],
        run_name=args.run_name,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_dim_256_cosine_ndcg@10",
        greater_is_better=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=relevance_dataset.select_columns(
            ["anchor", "positive", "negative"]
        ),
        loss=train_loss,
        evaluator=evaluator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving best model...")
    trainer.save_model(args.output_dir)

    print("Reloading fine-tuned model...")
    fine_tuned_model = SentenceTransformer(
        args.output_dir,
        device=device,
    )

    print("Evaluating fine-tuned model...")
    final_results = evaluator(fine_tuned_model)
    print_eval_results(final_results, matryoshka_dimensions)

    print("Running example encoding...")
    sentences = [
        "የተደጋገመው የመሬት መንቀጥቀጥና የእሳተ ገሞራ ምልክት በአፋር ክልል",
        "በአክሱም ከተማ የሚገኙ ሙስሊም ሴት ተማሪዎች ከሒጃብ መልበስ ጋር በተያያዘ ውዝግብ ከትምህርት ገበታ ውጭ ሆነው እንደሚገኙ የትግራይ እስልምና ጉዳዮች ምክርቤት ስታወቀ።",
        "በማዕከላዊ ኢትዮጵያ ክልል ሃድያ ዞን ጊቤ ወረዳ በሚገኙ 12 ቀበሌዎች መሠረታዊ የመንግሥት አገልግሎት መስጫ ተቋማት በሙሉና በከፊል በመዘጋታቸው መቸገራቸውን ነዋሪዎች አመለከቱ።",
    ]

    embeddings = fine_tuned_model.encode(sentences)
    print(f"Embedding shape: {embeddings.shape}")

    similarities = fine_tuned_model.similarity(embeddings, embeddings)
    print("Similarities:")
    print(similarities)

    print("Done.")


if __name__ == "__main__":
    main()
