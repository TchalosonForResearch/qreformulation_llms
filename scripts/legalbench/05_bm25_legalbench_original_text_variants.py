"""Compare BM25 indexing variants for the original LegalBench queries."""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm


DATA_DIR = Path("data/processed/legalbench/rag_mini")
RUNS_DIR = Path("runs/legalbench")
OUT_DIR = Path("outputs/tables/legalbench")

RUNS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


TEXT_VARIANTS = [
    "chunk_only",
    "filename_plus_chunk",
    "filepath_plus_chunk",
    "task_filename_plus_chunk",
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def simple_tokenize(text: str) -> list[str]:
    """
    Tokenisation simple pour BM25.
    """
    text = text.lower()
    return re.findall(r"\b\w+\b", text, flags=re.UNICODE)


def clean_metadata_text(text: str) -> str:
    """
    Transforme un nom de fichier ou chemin en texte plus lisible pour BM25.

    Exemple :
      "maud/The Michaels Companies, Inc._Apollo Global Management, LLC.txt"

    devient quelque chose comme :
      "maud The Michaels Companies Inc Apollo Global Management LLC"
    """
    text = text.replace("\\", "/")
    text = text.replace("/", " ")
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace(".txt", " ")
    text = text.replace(".pdf", " ")
    text = text.replace("||", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_filename(file_path: str) -> str:
    return Path(file_path.replace("\\", "/")).name


def build_index_text(doc: dict, variant: str) -> str:
    """
    Construit le texte réellement indexé par BM25 selon la variante.
    """
    chunk_text = doc.get("text", "") or ""
    file_path = str(doc.get("file_path", "") or "")
    task = str(doc.get("task", "") or "")

    filename = get_filename(file_path)

    filename_text = clean_metadata_text(filename)
    filepath_text = clean_metadata_text(file_path)
    task_text = clean_metadata_text(task)

    if variant == "chunk_only":
        return chunk_text

    if variant == "filename_plus_chunk":
        return f"{filename_text}\n{chunk_text}"

    if variant == "filepath_plus_chunk":
        return f"{filepath_text}\n{chunk_text}"

    if variant == "task_filename_plus_chunk":
        return f"{task_text}\n{filename_text}\n{chunk_text}"

    raise ValueError(f"Unknown text variant: {variant}")


def load_qrels(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["query_id", "iter", "doc_id", "relevance"],
        dtype={"query_id": str, "doc_id": str, "relevance": int},
    )


def build_qrels_dict(qrels_df: pd.DataFrame) -> dict[str, set[str]]:
    qrels = {}

    for qid, group in qrels_df.groupby("query_id"):
        relevant_docs = set(
            group.loc[group["relevance"] > 0, "doc_id"].astype(str)
        )
        qrels[str(qid)] = relevant_docs

    return qrels


def evaluate_run_per_query(
    run_df: pd.DataFrame,
    qrels: dict[str, set[str]],
    query_task: dict[str, str],
) -> pd.DataFrame:
    grouped = run_df.groupby("query_id", sort=False)

    rows = []

    for qid, relevant_docs in qrels.items():
        if qid in grouped.groups:
            ranked_docs = (
                grouped.get_group(qid)
                .sort_values("rank")["doc_id"]
                .astype(str)
                .tolist()
            )
        else:
            ranked_docs = []

        n_relevant = len(relevant_docs)

        retrieved_10 = set(ranked_docs[:10])
        retrieved_100 = set(ranked_docs[:100])

        recall_10 = (
            len(retrieved_10.intersection(relevant_docs)) / n_relevant
            if n_relevant > 0
            else 0.0
        )

        recall_100 = (
            len(retrieved_100.intersection(relevant_docs)) / n_relevant
            if n_relevant > 0
            else 0.0
        )

        mrr_10 = 0.0
        first_relevant_rank = None

        for rank, doc_id in enumerate(ranked_docs[:10], start=1):
            if doc_id in relevant_docs:
                mrr_10 = 1.0 / rank
                first_relevant_rank = rank
                break

        dcg_10 = 0.0

        for rank, doc_id in enumerate(ranked_docs[:10], start=1):
            if doc_id in relevant_docs:
                dcg_10 += 1.0 / np.log2(rank + 1)

        idcg_10 = sum(
            1.0 / np.log2(rank + 1)
            for rank in range(1, min(n_relevant, 10) + 1)
        )

        ndcg_10 = dcg_10 / idcg_10 if idcg_10 > 0 else 0.0

        rows.append(
            {
                "query_id": qid,
                "task": query_task.get(qid, "unknown"),
                "num_relevant": n_relevant,
                "Recall@10": recall_10,
                "Recall@100": recall_100,
                "MRR@10": mrr_10,
                "nDCG@10": ndcg_10,
                "first_relevant_rank": first_relevant_rank,
            }
        )

    return pd.DataFrame(rows)


def summarize_global(per_query_df: pd.DataFrame) -> dict:
    return {
        "num_queries": int(len(per_query_df)),
        "Recall@10": float(per_query_df["Recall@10"].mean()),
        "Recall@100": float(per_query_df["Recall@100"].mean()),
        "MRR@10": float(per_query_df["MRR@10"].mean()),
        "nDCG@10": float(per_query_df["nDCG@10"].mean()),
    }


def summarize_by_task(per_query_df: pd.DataFrame) -> pd.DataFrame:
    return (
        per_query_df
        .groupby("task")
        .agg(
            num_queries=("query_id", "count"),
            mean_num_relevant=("num_relevant", "mean"),
            **{
                "Recall@10": ("Recall@10", "mean"),
                "Recall@100": ("Recall@100", "mean"),
                "MRR@10": ("MRR@10", "mean"),
                "nDCG@10": ("nDCG@10", "mean"),
            },
        )
        .reset_index()
    )


def run_bm25_variant(
    *,
    variant: str,
    corpus: list[dict],
    queries: list[dict],
    qrels: dict[str, set[str]],
    query_task: dict[str, str],
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Lance BM25 pour une variante de texte indexé.
    """
    print("\n" + "=" * 80)
    print(f"Running BM25 variant: {variant}")
    print("=" * 80)

    doc_ids = [str(doc["doc_id"]) for doc in corpus]

    print("Building index texts...")
    index_texts = [
        build_index_text(doc, variant)
        for doc in tqdm(corpus, desc=f"Texts {variant}")
    ]

    print("Tokenizing corpus...")
    tokenized_corpus = [
        simple_tokenize(text)
        for text in tqdm(index_texts, desc=f"Tokenizing {variant}")
    ]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    run_rows = []
    effective_top_k = min(top_k, len(corpus))

    print("Retrieving...")
    for query in tqdm(queries, desc=f"Queries {variant}"):
        qid = str(query["query_id"])
        query_text = query.get("text", "") or ""
        tokenized_query = simple_tokenize(query_text)

        scores = bm25.get_scores(tokenized_query)

        if effective_top_k < len(scores):
            candidate_indices = np.argpartition(-scores, effective_top_k - 1)[:effective_top_k]
        else:
            candidate_indices = np.arange(len(scores))

        ranked_indices = sorted(
            candidate_indices,
            key=lambda idx: (-float(scores[idx]), doc_ids[idx]),
        )

        for rank, idx in enumerate(ranked_indices, start=1):
            run_rows.append(
                {
                    "query_id": qid,
                    "doc_id": doc_ids[idx],
                    "rank": rank,
                    "score": float(scores[idx]),
                    "method": f"bm25_original_{variant}",
                    "text_variant": variant,
                }
            )

    run_df = pd.DataFrame(run_rows)

    run_path = RUNS_DIR / f"bm25_original_{variant}_mini.tsv"
    run_df.to_csv(run_path, sep="\t", index=False)

    print(f"Saved run to: {run_path}")

    print("Evaluating...")
    per_query_df = evaluate_run_per_query(
        run_df=run_df,
        qrels=qrels,
        query_task=query_task,
    )

    per_query_df.insert(0, "method", f"bm25_original_{variant}")
    per_query_df.insert(1, "text_variant", variant)

    global_metrics = {
        "method": f"bm25_original_{variant}",
        "dataset": "legalbench_rag_mini",
        "text_variant": variant,
        **summarize_global(per_query_df),
    }

    global_df = pd.DataFrame([global_metrics])

    by_task_df = summarize_by_task(per_query_df)
    by_task_df.insert(0, "method", f"bm25_original_{variant}")
    by_task_df.insert(1, "text_variant", variant)

    print("\nGlobal metrics:")
    print(global_df.to_string(index=False))

    print("\nMetrics by task:")
    print(by_task_df.to_string(index=False))

    print("\nWorst 10 queries by nDCG@10:")
    worst = per_query_df.sort_values(
        ["nDCG@10", "Recall@100", "MRR@10"],
        ascending=[True, True, True],
    ).head(10)

    print(worst.to_string(index=False))

    # Libération mémoire entre variantes.
    del index_texts
    del tokenized_corpus
    del bm25
    gc.collect()

    return global_df, by_task_df, per_query_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare BM25 text variants on LegalBench-RAG mini."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Number of retrieved chunks saved per query.",
    )

    parser.add_argument(
        "--variants",
        nargs="*",
        default=TEXT_VARIANTS,
        choices=TEXT_VARIANTS,
        help="Text variants to run.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    corpus_path = DATA_DIR / "corpus.jsonl"
    queries_path = DATA_DIR / "queries.jsonl"
    qrels_path = DATA_DIR / "qrels.tsv"

    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus file: {corpus_path}")

    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")

    if not qrels_path.exists():
        raise FileNotFoundError(f"Missing qrels file: {qrels_path}")

    print("=" * 80)
    print("BM25 original text variants — LegalBench-RAG mini")
    print("=" * 80)

    print(f"Loading corpus: {corpus_path}")
    corpus = read_jsonl(corpus_path)

    print(f"Loading queries: {queries_path}")
    queries = read_jsonl(queries_path)

    print(f"Loading qrels: {qrels_path}")
    qrels_df = load_qrels(qrels_path)
    qrels = build_qrels_dict(qrels_df)

    query_task = {
        str(query["query_id"]): str(query.get("task", "unknown"))
        for query in queries
    }

    print(f"Corpus chunks: {len(corpus)}")
    print(f"Queries: {len(queries)}")
    print(f"Qrels queries: {len(qrels)}")
    print(f"Qrels rows: {len(qrels_df)}")
    print(f"Variants: {args.variants}")
    print(f"Top-k: {args.top_k}")

    all_global = []
    all_by_task = []
    all_per_query = []

    for variant in args.variants:
        global_df, by_task_df, per_query_df = run_bm25_variant(
            variant=variant,
            corpus=corpus,
            queries=queries,
            qrels=qrels,
            query_task=query_task,
            top_k=args.top_k,
        )

        all_global.append(global_df)
        all_by_task.append(by_task_df)
        all_per_query.append(per_query_df)

    metrics_df = pd.concat(all_global, ignore_index=True)
    by_task_all_df = pd.concat(all_by_task, ignore_index=True)
    per_query_all_df = pd.concat(all_per_query, ignore_index=True)

    metrics_path = OUT_DIR / "bm25_original_text_variants_mini_metrics.csv"
    by_task_path = OUT_DIR / "bm25_original_text_variants_mini_by_task.csv"
    per_query_path = OUT_DIR / "bm25_original_text_variants_mini_per_query.csv"

    metrics_df.to_csv(metrics_path, index=False)
    by_task_all_df.to_csv(by_task_path, index=False)
    per_query_all_df.to_csv(per_query_path, index=False)

    print("\n" + "=" * 80)
    print("All BM25 text variant metrics:")
    print(metrics_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("All BM25 text variant metrics by task:")
    print(by_task_all_df.to_string(index=False))

    print("\nSaved files:")
    print(metrics_path)
    print(by_task_path)
    print(per_query_path)

    print("\nNext decision:")
    print("  Choose the canonical LegalBench BM25 baseline variant.")
    print("  Then continue with reformulations and original-preserving fusion.")


if __name__ == "__main__":
    main()