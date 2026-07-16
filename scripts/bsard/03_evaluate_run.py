"""Evaluate a BSARD retrieval run against the benchmark qrels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("outputs/tables/bsard")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_qrels(path: Path) -> pd.DataFrame:
    """
    Charge les jugements de pertinence.

    Format attendu :
        query_id    iter    doc_id    relevance

    Les fichiers qrels de BSARD n'ont pas de vraie ligne d'en-tête.
    On impose donc header=None.
    """
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["query_id", "iter", "doc_id", "relevance"],
        dtype={"query_id": str, "doc_id": str, "relevance": int},
    )


def build_qrels_dict(qrels_df: pd.DataFrame) -> dict[str, set[str]]:
    """
    Transforme les qrels en dictionnaire :
        query_id -> ensemble des documents pertinents.
    """
    qrels = {}

    for qid, group in qrels_df.groupby("query_id"):
        relevant_docs = set(
            group.loc[group["relevance"] > 0, "doc_id"].astype(str)
        )
        qrels[str(qid)] = relevant_docs

    return qrels


def load_run(path: Path) -> pd.DataFrame:
    """
    Charge un fichier run.

    Notre premier run est en TSV avec header :
        query_id, doc_id, rank, score, method

    Plus tard, certains runs pourront être au format TREC :
        query_id Q0 doc_id rank score run_name

    Cette fonction essaie de supporter les deux formats.
    """
    if not path.exists():
        raise FileNotFoundError(f"Run file not found: {path}")

    # On essaie d'abord le format TSV avec header.
    df = pd.read_csv(path, sep="\t", dtype=str)

    expected_columns = {"query_id", "doc_id", "rank", "score"}

    if expected_columns.issubset(set(df.columns)):
        df["rank"] = df["rank"].astype(int)
        df["score"] = df["score"].astype(float)
        return df

    # Si le format avec header ne marche pas, on essaie le format TREC sans header.
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["query_id", "Q0", "doc_id", "rank", "score", "run_name"],
        dtype={"query_id": str, "doc_id": str, "run_name": str},
        engine="python",
    )

    df["rank"] = df["rank"].astype(int)
    df["score"] = df["score"].astype(float)
    return df


def compute_query_metrics(
    qid: str,
    ranked_docs: list[str],
    relevant_docs: set[str],
) -> dict:
    """
    Calcule les métriques pour une seule requête.
    """
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

    return {
        "query_id": qid,
        "num_relevant": n_relevant,
        "Recall@10": recall_10,
        "Recall@100": recall_100,
        "MRR@10": mrr_10,
        "nDCG@10": ndcg_10,
        "first_relevant_rank": first_relevant_rank,
    }


def evaluate(run_df: pd.DataFrame, qrels: dict[str, set[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produit :
    - une table per-query ;
    - une table globale.
    """
    grouped_run = run_df.groupby("query_id", sort=False)

    per_query_rows = []

    for qid, relevant_docs in qrels.items():
        if qid in grouped_run.groups:
            ranked_docs = (
                grouped_run.get_group(qid)
                .sort_values("rank")["doc_id"]
                .astype(str)
                .tolist()
            )
        else:
            ranked_docs = []

        per_query_rows.append(
            compute_query_metrics(
                qid=qid,
                ranked_docs=ranked_docs,
                relevant_docs=relevant_docs,
            )
        )

    per_query_df = pd.DataFrame(per_query_rows)

    metric_columns = ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]

    global_df = pd.DataFrame(
        [
            {
                "num_queries": len(per_query_df),
                **{
                    metric: float(per_query_df[metric].mean())
                    for metric in metric_columns
                },
            }
        ]
    )

    return per_query_df, global_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a BSARD retrieval run globally and per query."
    )

    parser.add_argument(
        "--run",
        required=True,
        type=Path,
        help="Path to the run file.",
    )

    parser.add_argument(
        "--qrels",
        required=True,
        type=Path,
        help="Path to the qrels file.",
    )

    parser.add_argument(
        "--name",
        required=True,
        type=str,
        help="Name used to save output files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading run: {args.run}")
    run_df = load_run(args.run)

    print(f"Loading qrels: {args.qrels}")
    qrels_df = load_qrels(args.qrels)
    qrels = build_qrels_dict(qrels_df)

    print(f"Run rows: {len(run_df)}")
    print(f"Queries in qrels: {len(qrels)}")

    per_query_df, global_df = evaluate(run_df, qrels)

    per_query_path = OUT_DIR / f"{args.name}_per_query.csv"
    metrics_path = OUT_DIR / f"{args.name}_metrics.csv"

    per_query_df.to_csv(per_query_path, index=False)
    global_df.to_csv(metrics_path, index=False)

    print("\nGlobal metrics:")
    print(global_df.to_string(index=False))

    print(f"\nSaved global metrics to: {metrics_path}")
    print(f"Saved per-query metrics to: {per_query_path}")

    print("\nTen worst queries by nDCG@10:")
    print(
        per_query_df.sort_values("nDCG@10")
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()