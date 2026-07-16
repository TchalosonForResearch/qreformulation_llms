"""Run the original-query BM25 baseline on BSARD."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm


RAW_DIR = Path("data/raw/bsard")
RUNS_DIR = Path("runs/bsard")
OUT_DIR = Path("outputs/tables/bsard")

RUNS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


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

    On met le texte en minuscules, puis on extrait les mots.
    BM25 compare ensuite les mots de la requête avec les mots
    des articles.
    """
    text = text.lower()
    return re.findall(r"\b\w+\b", text, flags=re.UNICODE)


def load_qrels(path: Path) -> pd.DataFrame:
    """
    Charge les qrels BSARD.

    Format TREC :
        query_id    iter    doc_id    relevance

    Les fichiers qrels n'ont pas de vraie ligne d'en-tête.
    """
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


def evaluate_run(run_df: pd.DataFrame, qrels: dict[str, set[str]]) -> dict[str, float]:
    """
    Calcule les métriques globales :
    Recall@10, Recall@100, MRR@10, nDCG@10.
    """
    metrics = {
        "Recall@10": [],
        "Recall@100": [],
        "MRR@10": [],
        "nDCG@10": [],
    }

    grouped = run_df.groupby("query_id", sort=False)

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

        rr = 0.0
        for rank, doc_id in enumerate(ranked_docs[:10], start=1):
            if doc_id in relevant_docs:
                rr = 1.0 / rank
                break

        dcg = 0.0
        for rank, doc_id in enumerate(ranked_docs[:10], start=1):
            if doc_id in relevant_docs:
                dcg += 1.0 / np.log2(rank + 1)

        idcg = sum(
            1.0 / np.log2(rank + 1)
            for rank in range(1, min(n_relevant, 10) + 1)
        )

        ndcg = dcg / idcg if idcg > 0 else 0.0

        metrics["Recall@10"].append(recall_10)
        metrics["Recall@100"].append(recall_100)
        metrics["MRR@10"].append(rr)
        metrics["nDCG@10"].append(ndcg)

    return {name: float(np.mean(values)) for name, values in metrics.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simple BM25 original baseline on BSARD."
    )

    parser.add_argument(
        "--split",
        choices=["train", "test"],
        required=True,
        help="Which BSARD split to run: train or test.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Number of documents retrieved per query.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = args.split

    corpus_path = RAW_DIR / "corpus.jsonl"
    queries_path = RAW_DIR / f"queries_{split}.jsonl"
    qrels_path = RAW_DIR / f"qrels_{split}.tsv"

    for path in [corpus_path, queries_path, qrels_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    print(f"Running BM25 original on BSARD split: {split}")

    print("Loading corpus...")
    corpus = read_jsonl(corpus_path)

    print("Loading queries...")
    queries = read_jsonl(queries_path)

    print("Loading qrels...")
    qrels_df = load_qrels(qrels_path)
    qrels = build_qrels_dict(qrels_df)

    print(f"Corpus documents: {len(corpus)}")
    print(f"Queries: {len(queries)}")
    print(f"Qrels queries: {len(qrels)}")

    doc_ids = [str(doc["doc_id"]) for doc in corpus]
    doc_texts = [doc.get("text", "") or "" for doc in corpus]

    print("Tokenizing corpus...")
    tokenized_corpus = [simple_tokenize(text) for text in tqdm(doc_texts)]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    run_rows = []

    print("Retrieving...")
    for query in tqdm(queries):
        qid = str(query["query_id"])
        query_text = query.get("text") or query.get("question") or ""

        tokenized_query = simple_tokenize(query_text)
        scores = bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][: args.top_k]

        for rank, idx in enumerate(top_indices, start=1):
            run_rows.append(
                {
                    "query_id": qid,
                    "doc_id": doc_ids[idx],
                    "rank": rank,
                    "score": float(scores[idx]),
                    "method": "bm25_original",
                }
            )

    run_df = pd.DataFrame(run_rows)

    run_path = RUNS_DIR / f"bm25_original_{split}.tsv"
    run_df.to_csv(run_path, sep="\t", index=False)

    print(f"Saved run to: {run_path}")

    metrics = evaluate_run(run_df, qrels)
    metrics_df = pd.DataFrame(
        [
            {
                "method": "bm25_original",
                "split": split,
                **metrics,
            }
        ]
    )

    metrics_path = OUT_DIR / f"bm25_original_{split}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()