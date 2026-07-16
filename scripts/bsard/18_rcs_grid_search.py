"""Search the exploratory BSARD RCS parameter grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RUNS_DIR = Path("runs/bsard")
OUT_DIR = Path("outputs/tables/bsard")
GRID_RUNS_DIR = RUNS_DIR / "rcs_grid"
QRELS_PATH = Path("data/raw/bsard/qrels_test.tsv")

OUT_DIR.mkdir(parents=True, exist_ok=True)
GRID_RUNS_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_RUN = RUNS_DIR / "bm25_original_test.tsv"

GROUPS = [
    {
        "group_name": "deepseek",
        "fusion_type": "rcs_grid_deepseek",
        "reformulation_runs": [
            RUNS_DIR / "bm25_deepseek_legal_rewrite_test.tsv",
            RUNS_DIR / "bm25_deepseek_keyword_expansion_test.tsv",
            RUNS_DIR / "bm25_deepseek_hyde_style_test.tsv",
        ],
    },
    {
        "group_name": "gpt",
        "fusion_type": "rcs_grid_gpt",
        "reformulation_runs": [
            RUNS_DIR / "bm25_gpt_legal_rewrite_test.tsv",
            RUNS_DIR / "bm25_gpt_keyword_expansion_test.tsv",
            RUNS_DIR / "bm25_gpt_hyde_style_test.tsv",
        ],
    },
    {
        "group_name": "deepseek+gpt",
        "fusion_type": "rcs_grid_all_generators",
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

ALPHA_VALUES = [1.0]
BETA_VALUES = [0.5, 1.0]
GAMMA_VALUES = [0.0, 0.5, 1.0, 2.0]
MIN_VOTES_VALUES = [2, 3]

TOP_K = 1000
RRF_K = 60

METRICS = ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]


def load_run(path: Path, top_k: int = TOP_K) -> pd.DataFrame:
    """
    Charge un run TSV et applique les mêmes protections que nos scripts RRF :
    - top_k ;
    - suppression des doublons query_id/doc_id ;
    - meilleur rang conservé.
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
    result = {}

    for qid, group in run_df.groupby("query_id"):
        result[str(qid)] = {
            str(row["doc_id"]): int(row["rank"])
            for _, row in group.iterrows()
        }

    return result


def evaluate_run(run_df: pd.DataFrame, qrels: dict[str, set[str]]) -> dict[str, float]:
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


def compute_rcs_run(
    *,
    original_ranks: dict[str, dict[str, int]],
    reform_ranks_list: list[dict[str, dict[str, int]]],
    run_name: str,
    alpha: float,
    beta: float,
    gamma: float,
    min_votes: int,
    rrf_k: int = RRF_K,
    top_k: int = TOP_K,
) -> pd.DataFrame:
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
                reform_rank = reform_ranks.get(qid, {}).get(doc_id)

                if reform_rank is not None:
                    reform_score += 1.0 / (rrf_k + reform_rank)
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

        ranked = sorted(
            scored_docs,
            key=lambda row: (-row["score"], row["doc_id"]),
        )[:top_k]

        for rank, row in enumerate(ranked, start=1):
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
        description="Run RCS parameter grid search on BSARD."
    )

    parser.add_argument(
        "--save-runs",
        action="store_true",
        help="Save every RCS run file. By default, only metrics are saved.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading qrels...")
    qrels = build_qrels_dict(load_qrels(QRELS_PATH))

    print("Loading original run...")
    original_run = load_run(ORIGINAL_RUN, top_k=TOP_K)
    original_ranks = run_to_rank_dict(original_run)
    baseline_metrics = evaluate_run(original_run, qrels)

    print("\nBaseline BM25 original:")
    print(pd.DataFrame([{"method": "bm25_original_test", **baseline_metrics}]).to_string(index=False))

    # Préchargement des reformulation ranks pour éviter de relire les fichiers.
    print("\nPreloading reformulation runs...")
    preloaded_group_ranks = {}

    for group in GROUPS:
        ranks_list = []

        for path in group["reformulation_runs"]:
            run_df = load_run(path, top_k=TOP_K)
            ranks_list.append(run_to_rank_dict(run_df))

        preloaded_group_ranks[group["group_name"]] = ranks_list
        print(f"Loaded group {group['group_name']} with {len(ranks_list)} reformulation runs")

    rows = []

    total_configs = (
        len(GROUPS)
        * len(ALPHA_VALUES)
        * len(BETA_VALUES)
        * len(GAMMA_VALUES)
        * len(MIN_VOTES_VALUES)
    )

    done = 0

    for group in GROUPS:
        group_name = group["group_name"]
        fusion_type = group["fusion_type"]
        reform_ranks_list = preloaded_group_ranks[group_name]

        for alpha in ALPHA_VALUES:
            for beta in BETA_VALUES:
                for gamma in GAMMA_VALUES:
                    for min_votes in MIN_VOTES_VALUES:
                        done += 1

                        run_name = (
                            f"rcs_grid_{group_name}"
                            f"_a{alpha:g}_b{beta:g}_g{gamma:g}_v{min_votes}"
                        ).replace("+", "plus").replace(".", "p")

                        print(
                            f"[{done}/{total_configs}] "
                            f"group={group_name}, alpha={alpha}, beta={beta}, "
                            f"gamma={gamma}, min_votes={min_votes}"
                        )

                        rcs_run = compute_rcs_run(
                            original_ranks=original_ranks,
                            reform_ranks_list=reform_ranks_list,
                            run_name=run_name,
                            alpha=alpha,
                            beta=beta,
                            gamma=gamma,
                            min_votes=min_votes,
                            rrf_k=RRF_K,
                            top_k=TOP_K,
                        )

                        metrics = evaluate_run(rcs_run, qrels)

                        row = {
                            "method": run_name,
                            "group": group_name,
                            "fusion_type": fusion_type,
                            "num_reformulations": len(reform_ranks_list),
                            "rrf_k": RRF_K,
                            "top_k": TOP_K,
                            "alpha": alpha,
                            "beta": beta,
                            "gamma": gamma,
                            "min_votes": min_votes,
                            **metrics,
                        }

                        # Gains par rapport à BM25 original
                        for metric in METRICS:
                            row[f"delta_{metric}"] = metrics[metric] - baseline_metrics[metric]

                        rows.append(row)

                        if args.save_runs:
                            run_path = GRID_RUNS_DIR / f"{run_name}.tsv"
                            rcs_run.to_csv(run_path, sep="\t", index=False)

    results_df = pd.DataFrame(rows)

    metrics_path = OUT_DIR / "rcs_grid_search_metrics.csv"
    results_df.to_csv(metrics_path, index=False)

    # Meilleure configuration par métrique et par groupe.
    best_rows = []

    for group_name, group_df in results_df.groupby("group"):
        for metric in METRICS:
            best = group_df.sort_values(metric, ascending=False).iloc[0]
            best_rows.append(
                {
                    "group": group_name,
                    "best_for_metric": metric,
                    "method": best["method"],
                    "alpha": best["alpha"],
                    "beta": best["beta"],
                    "gamma": best["gamma"],
                    "min_votes": best["min_votes"],
                    metric: best[metric],
                    f"delta_{metric}": best[f"delta_{metric}"],
                    "Recall@10": best["Recall@10"],
                    "Recall@100": best["Recall@100"],
                    "MRR@10": best["MRR@10"],
                    "nDCG@10": best["nDCG@10"],
                }
            )

    best_df = pd.DataFrame(best_rows)
    best_path = OUT_DIR / "rcs_grid_search_best_by_metric.csv"
    best_df.to_csv(best_path, index=False)

    print("\n" + "=" * 80)
    print("Top 10 configurations by nDCG@10:")
    print(
        results_df.sort_values("nDCG@10", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("Top 10 configurations by Recall@100:")
    print(
        results_df.sort_values("Recall@100", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("Best configuration by metric and group:")
    print(best_df.to_string(index=False))

    print("\nSaved files:")
    print(metrics_path)
    print(best_path)

    if args.save_runs:
        print(f"Saved grid runs to: {GRID_RUNS_DIR}")


if __name__ == "__main__":
    main()