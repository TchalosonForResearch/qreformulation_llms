"""Build the exploratory Retrieval Consensus Score runs for BSARD."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RUNS_DIR = Path("runs/bsard")
OUT_DIR = Path("outputs/tables/bsard")
QRELS_PATH = Path("data/raw/bsard/qrels_test.tsv")

OUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_RUN = RUNS_DIR / "bm25_original_test.tsv"

RCS_GROUPS = [
    {
        "name": "rcs_deepseek_test",
        "generator": "deepseek",
        "fusion_type": "rcs_original_plus_all_deepseek",
        "reformulation_runs": [
            RUNS_DIR / "bm25_deepseek_legal_rewrite_test.tsv",
            RUNS_DIR / "bm25_deepseek_keyword_expansion_test.tsv",
            RUNS_DIR / "bm25_deepseek_hyde_style_test.tsv",
        ],
    },
    {
        "name": "rcs_gpt_test",
        "generator": "gpt",
        "fusion_type": "rcs_original_plus_all_gpt",
        "reformulation_runs": [
            RUNS_DIR / "bm25_gpt_legal_rewrite_test.tsv",
            RUNS_DIR / "bm25_gpt_keyword_expansion_test.tsv",
            RUNS_DIR / "bm25_gpt_hyde_style_test.tsv",
        ],
    },
    {
        "name": "rcs_all_generators_test",
        "generator": "deepseek+gpt",
        "fusion_type": "rcs_original_plus_all_generators",
        "reformulation_runs": [
            RUNS_DIR / "bm25_deepseek_legal_rewrite_test.tsv",
            RUNS_DIR / "bm25_deepseek_keyword_expansion_test.tsv",
            RUNS_DIR / "bm25_deepseek_hyde_style_test.tsv",
            RUNS_DIR / "bm25_gpt_legal_rewrite_test.tsv",
            RUNS_DIR / "bm25_gpt_keyword_expansion_test.tsv",
            RUNS_DIR / "bm25_gpt_hyde_style_test.tsv",
        ],
    },
]


def load_run(path: Path, top_k: int) -> pd.DataFrame:
    """
    Charge un run TSV.

    Nettoyages :
    - garde seulement les documents jusqu'au rang top_k ;
    - supprime les doublons query_id/doc_id ;
    - garde le meilleur rang.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing run file: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype={"query_id": str, "doc_id": str},
    )

    required = {"query_id", "doc_id", "rank", "score"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df["rank"] = df["rank"].astype(int)
    df["score"] = df["score"].astype(float)

    df = df[df["rank"] <= top_k].copy()

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
        qrels[str(qid)] = set(
            group.loc[group["relevance"] > 0, "doc_id"].astype(str)
        )

    return qrels


def run_to_rank_dict(run_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """
    Convertit un run en dictionnaire :
        query_id -> doc_id -> rank
    """
    result = {}

    for qid, group in run_df.groupby("query_id"):
        result[str(qid)] = {
            str(row["doc_id"]): int(row["rank"])
            for _, row in group.iterrows()
        }

    return result


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

        for cutoff in [10, 100]:
            retrieved = set(ranked_docs[:cutoff])
            recall = (
                len(retrieved.intersection(relevant_docs)) / n_relevant
                if n_relevant > 0
                else 0.0
            )
            metrics[f"Recall@{cutoff}"].append(recall)

        rr = 0.0
        for rank, doc_id in enumerate(ranked_docs[:10], start=1):
            if doc_id in relevant_docs:
                rr = 1.0 / rank
                break

        metrics["MRR@10"].append(rr)

        dcg = 0.0
        for rank, doc_id in enumerate(ranked_docs[:10], start=1):
            if doc_id in relevant_docs:
                dcg += 1.0 / np.log2(rank + 1)

        idcg = sum(
            1.0 / np.log2(rank + 1)
            for rank in range(1, min(n_relevant, 10) + 1)
        )

        metrics["nDCG@10"].append(dcg / idcg if idcg > 0 else 0.0)

    return {metric: float(np.mean(values)) for metric, values in metrics.items()}


def compute_rcs(
    *,
    original_ranks: dict[str, dict[str, int]],
    reform_ranks_list: list[dict[str, dict[str, int]]],
    run_name: str,
    rrf_k: int,
    top_k: int,
    min_votes: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> pd.DataFrame:
    """
    Calcule le score RCS query par query.
    """
    all_query_ids = set(original_ranks.keys())

    for reform_ranks in reform_ranks_list:
        all_query_ids.update(reform_ranks.keys())

    fused_rows = []
    num_reforms = len(reform_ranks_list)

    for qid in sorted(all_query_ids, key=lambda x: int(x) if x.isdigit() else x):
        candidate_docs = set()

        anchor_docs = original_ranks.get(qid, {})
        candidate_docs.update(anchor_docs.keys())

        for reform_ranks in reform_ranks_list:
            candidate_docs.update(reform_ranks.get(qid, {}).keys())

        scored_docs = []

        for doc_id in candidate_docs:
            anchor_rank = anchor_docs.get(doc_id)

            if anchor_rank is None:
                anchor_score = 0.0
                anchor_present = False
            else:
                anchor_score = 1.0 / (rrf_k + anchor_rank)
                anchor_present = True

            reform_score = 0.0
            votes = 0

            for reform_ranks in reform_ranks_list:
                rank = reform_ranks.get(qid, {}).get(doc_id)

                if rank is not None:
                    reform_score += 1.0 / (rrf_k + rank)
                    votes += 1

            if anchor_present and votes >= min_votes and num_reforms > 0:
                consensus_bonus = anchor_score * (votes / num_reforms)
            else:
                consensus_bonus = 0.0

            final_score = (
                alpha * anchor_score
                + beta * reform_score
                + gamma * consensus_bonus
            )

            scored_docs.append(
                {
                    "query_id": qid,
                    "doc_id": doc_id,
                    "score": final_score,
                    "anchor_rank": anchor_rank,
                    "anchor_score": anchor_score,
                    "reform_score": reform_score,
                    "consensus_votes": votes,
                    "consensus_bonus": consensus_bonus,
                }
            )

        ranked_docs = sorted(
            scored_docs,
            key=lambda row: (-row["score"], row["doc_id"]),
        )[:top_k]

        for rank, row in enumerate(ranked_docs, start=1):
            fused_rows.append(
                {
                    "query_id": row["query_id"],
                    "doc_id": row["doc_id"],
                    "rank": rank,
                    "score": row["score"],
                    "method": run_name,
                    "anchor_rank": row["anchor_rank"],
                    "anchor_score": row["anchor_score"],
                    "reform_score": row["reform_score"],
                    "consensus_votes": row["consensus_votes"],
                    "consensus_bonus": row["consensus_bonus"],
                }
            )

    return pd.DataFrame(fused_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simple RCS on BSARD BM25 runs."
    )

    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--min-votes", type=int, default=2)

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("RCS parameters:")
    print(f"  top_k     = {args.top_k}")
    print(f"  rrf_k     = {args.rrf_k}")
    print(f"  min_votes = {args.min_votes}")
    print(f"  alpha     = {args.alpha}")
    print(f"  beta      = {args.beta}")
    print(f"  gamma     = {args.gamma}")

    print("\nLoading qrels...")
    qrels = build_qrels_dict(load_qrels(QRELS_PATH))

    print("Loading original run...")
    original_run = load_run(ORIGINAL_RUN, top_k=args.top_k)
    original_ranks = run_to_rank_dict(original_run)

    baseline_metrics = evaluate_run(original_run, qrels)

    metrics_rows = [
        {
            "method": "bm25_original_test",
            "generator": "none",
            "fusion_type": "none",
            "num_reformulations": 0,
            "rrf_k": args.rrf_k,
            "min_votes": args.min_votes,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            **baseline_metrics,
        }
    ]

    print("\nBaseline:")
    print(pd.DataFrame([metrics_rows[0]]).to_string(index=False))

    for group in RCS_GROUPS:
        print("\n" + "=" * 80)
        print(f"Running {group['name']}")

        reform_ranks_list = []

        for path in group["reformulation_runs"]:
            run_df = load_run(path, top_k=args.top_k)
            reform_ranks_list.append(run_to_rank_dict(run_df))

        rcs_run = compute_rcs(
            original_ranks=original_ranks,
            reform_ranks_list=reform_ranks_list,
            run_name=group["name"],
            rrf_k=args.rrf_k,
            top_k=args.top_k,
            min_votes=args.min_votes,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
        )

        run_path = RUNS_DIR / f"{group['name']}.tsv"
        rcs_run.to_csv(run_path, sep="\t", index=False)

        metrics = evaluate_run(rcs_run, qrels)

        row = {
            "method": group["name"],
            "generator": group["generator"],
            "fusion_type": group["fusion_type"],
            "num_reformulations": len(group["reformulation_runs"]),
            "rrf_k": args.rrf_k,
            "min_votes": args.min_votes,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            **metrics,
        }

        metrics_rows.append(row)

        print(f"Saved RCS run to: {run_path}")
        print(pd.DataFrame([row]).to_string(index=False))

    metrics_df = pd.DataFrame(metrics_rows)

    metrics_path = OUT_DIR / "rcs_simple_test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("\n" + "=" * 80)
    print("RCS simple metrics:")
    print(metrics_df.to_string(index=False))

    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()