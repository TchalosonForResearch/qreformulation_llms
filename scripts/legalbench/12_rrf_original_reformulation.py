"""Fuse each LegalBench reformulation with the original query using RRF."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/processed/legalbench/rag_mini")
RUNS_DIR = Path("runs/legalbench")
OUT_DIR = Path("outputs/tables/legalbench")

RUNS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_RUN = RUNS_DIR / "bm25_original_mini_canonical.tsv"

GENERATORS = ["deepseek", "gpt"]
QUERY_TYPES = ["legal_rewrite", "keyword_expansion", "hyde_style"]

METRICS = ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {path}: {exc}"
                ) from exc

    return rows


def load_run(path: Path, top_k: int) -> pd.DataFrame:
    """
    Charge un fichier run TSV.

    Format attendu :
      query_id, doc_id, rank, score, ...
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing run file: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype={"query_id": str, "doc_id": str},
    )

    required = {"query_id", "doc_id", "rank"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Run file {path} missing columns: {missing}")

    df = df[["query_id", "doc_id", "rank"]].copy()
    df["rank"] = df["rank"].astype(int)

    # On limite explicitement les entrées RRF au top_k.
    df = df[df["rank"] <= top_k].copy()

    # Sécurité : si un doc apparaît plusieurs fois pour une même query,
    # on garde le meilleur rang.
    df = df.sort_values(["query_id", "doc_id", "rank"])
    df = df.drop_duplicates(["query_id", "doc_id"], keep="first")

    return df


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


def rrf_fuse_two_runs(
    *,
    original_df: pd.DataFrame,
    reform_df: pd.DataFrame,
    rrf_k: int,
    top_k: int,
) -> pd.DataFrame:
    """
    Fusionne deux runs avec Reciprocal Rank Fusion.

    score(doc) = 1/(rrf_k + rank_original) + 1/(rrf_k + rank_reformulation)
    """
    scores_by_qid: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for row in original_df.itertuples(index=False):
        qid = str(row.query_id)
        doc_id = str(row.doc_id)
        rank = int(row.rank)
        scores_by_qid[qid][doc_id] += 1.0 / (rrf_k + rank)

    for row in reform_df.itertuples(index=False):
        qid = str(row.query_id)
        doc_id = str(row.doc_id)
        rank = int(row.rank)
        scores_by_qid[qid][doc_id] += 1.0 / (rrf_k + rank)

    fused_rows = []

    for qid, doc_scores in scores_by_qid.items():
        ranked = sorted(
            doc_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]

        for rank, (doc_id, score) in enumerate(ranked, start=1):
            fused_rows.append(
                {
                    "query_id": qid,
                    "doc_id": doc_id,
                    "rank": rank,
                    "score": score,
                }
            )

    fused_df = pd.DataFrame(fused_rows)

    return fused_df


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


def evaluate_and_format(
    *,
    run_df: pd.DataFrame,
    method: str,
    generator: str,
    query_type: str,
    fusion: str,
    rrf_k: int | None,
    qrels: dict[str, set[str]],
    query_task: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_query_df = evaluate_run_per_query(
        run_df=run_df,
        qrels=qrels,
        query_task=query_task,
    )

    per_query_df.insert(0, "method", method)
    per_query_df.insert(1, "generator", generator)
    per_query_df.insert(2, "query_type", query_type)
    per_query_df.insert(3, "fusion", fusion)
    per_query_df.insert(4, "rrf_k", rrf_k)

    global_df = pd.DataFrame(
        [
            {
                "method": method,
                "dataset": "legalbench_rag_mini",
                "generator": generator,
                "query_type": query_type,
                "fusion": fusion,
                "rrf_k": rrf_k,
                **summarize_global(per_query_df),
            }
        ]
    )

    by_task_df = summarize_by_task(per_query_df)
    by_task_df.insert(0, "method", method)
    by_task_df.insert(1, "generator", generator)
    by_task_df.insert(2, "query_type", query_type)
    by_task_df.insert(3, "fusion", fusion)
    by_task_df.insert(4, "rrf_k", rrf_k)

    return global_df, by_task_df, per_query_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RRF original + one reformulation on LegalBench-RAG mini."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Maximum rank depth loaded and saved.",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF k constant.",
    )

    parser.add_argument(
        "--generators",
        nargs="*",
        default=GENERATORS,
        choices=GENERATORS,
        help="Generators to fuse.",
    )

    parser.add_argument(
        "--query-types",
        nargs="*",
        default=QUERY_TYPES,
        choices=QUERY_TYPES,
        help="Reformulation query types to fuse.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    queries_path = DATA_DIR / "queries.jsonl"
    qrels_path = DATA_DIR / "qrels.tsv"

    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")

    if not qrels_path.exists():
        raise FileNotFoundError(f"Missing qrels file: {qrels_path}")

    if not ORIGINAL_RUN.exists():
        raise FileNotFoundError(
            f"Missing canonical original run: {ORIGINAL_RUN}\n"
            "Run script 06_select_canonical_bm25_baseline.py first."
        )

    print("=" * 80)
    print("RRF original + reformulation — LegalBench-RAG mini")
    print("=" * 80)
    print(f"Original run: {ORIGINAL_RUN}")
    print(f"RRF k: {args.rrf_k}")
    print(f"Top-k: {args.top_k}")
    print(f"Generators: {args.generators}")
    print(f"Query types: {args.query_types}")

    queries = read_jsonl(queries_path)
    query_task = {
        str(query["query_id"]): str(query.get("task", "unknown"))
        for query in queries
    }

    qrels_df = load_qrels(qrels_path)
    qrels = build_qrels_dict(qrels_df)

    print("\nLoading canonical original run...")
    original_df = load_run(ORIGINAL_RUN, top_k=args.top_k)

    print(f"Original run rows loaded: {len(original_df)}")

    all_global = []
    all_by_task = []
    all_per_query = []

    print("\nEvaluating canonical original baseline for reference...")
    baseline_global, baseline_by_task, baseline_per_query = evaluate_and_format(
        run_df=original_df,
        method="bm25_original_mini_canonical",
        generator="none",
        query_type="original",
        fusion="none",
        rrf_k=None,
        qrels=qrels,
        query_task=query_task,
    )

    all_global.append(baseline_global)
    all_by_task.append(baseline_by_task)
    all_per_query.append(baseline_per_query)

    print("\nBaseline metrics:")
    print(baseline_global.to_string(index=False))

    for generator in args.generators:
        for query_type in args.query_types:
            reform_method = f"bm25_{generator}_{query_type}"
            reform_run = RUNS_DIR / f"{reform_method}_mini.tsv"

            print("\n" + "=" * 80)
            print(f"Fusion: original + {generator}/{query_type}")
            print("=" * 80)
            print(f"Reformulation run: {reform_run}")

            reform_df = load_run(reform_run, top_k=args.top_k)

            fused_df = rrf_fuse_two_runs(
                original_df=original_df,
                reform_df=reform_df,
                rrf_k=args.rrf_k,
                top_k=args.top_k,
            )

            fused_method = f"rrf_bm25_original_{generator}_{query_type}_mini"

            fused_df["method"] = fused_method
            fused_df["generator"] = generator
            fused_df["query_type"] = query_type
            fused_df["fusion"] = "original_plus_one_reformulation"
            fused_df["rrf_k"] = args.rrf_k

            fused_run_path = RUNS_DIR / f"{fused_method}.tsv"
            fused_df.to_csv(fused_run_path, sep="\t", index=False)

            print(f"Saved fused run to: {fused_run_path}")

            global_df, by_task_df, per_query_df = evaluate_and_format(
                run_df=fused_df,
                method=fused_method,
                generator=generator,
                query_type=query_type,
                fusion="original_plus_one_reformulation",
                rrf_k=args.rrf_k,
                qrels=qrels,
                query_task=query_task,
            )

            all_global.append(global_df)
            all_by_task.append(by_task_df)
            all_per_query.append(per_query_df)

            print("\nFusion metrics:")
            print(global_df.to_string(index=False))

    metrics_df = pd.concat(all_global, ignore_index=True)
    by_task_df = pd.concat(all_by_task, ignore_index=True)
    per_query_all_df = pd.concat(all_per_query, ignore_index=True)

    metrics_path = OUT_DIR / "rrf_original_reformulation_mini_metrics.csv"
    by_task_path = OUT_DIR / "rrf_original_reformulation_mini_by_task.csv"
    per_query_path = OUT_DIR / "rrf_original_reformulation_mini_per_query_all.csv"

    metrics_df.to_csv(metrics_path, index=False)
    by_task_df.to_csv(by_task_path, index=False)
    per_query_all_df.to_csv(per_query_path, index=False)

    print("\n" + "=" * 80)
    print("All RRF original + reformulation metrics")
    print("=" * 80)
    print(metrics_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("All RRF original + reformulation metrics by task")
    print("=" * 80)
    print(by_task_df.to_string(index=False))

    print("\nSaved files:")
    print(metrics_path)
    print(by_task_path)
    print(per_query_path)

    print("\nNext:")
    print("  scripts/legalbench/13_rrf_harm_rate_and_stats.py")


if __name__ == "__main__":
    main()