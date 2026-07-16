"""Build multi-view LegalBench RRF and exploratory RCS runs."""

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

METRICS = ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]


VIEW_RUNS = {
    "deepseek_legal_rewrite": RUNS_DIR / "bm25_deepseek_legal_rewrite_mini.tsv",
    "deepseek_keyword_expansion": RUNS_DIR / "bm25_deepseek_keyword_expansion_mini.tsv",
    "deepseek_hyde_style": RUNS_DIR / "bm25_deepseek_hyde_style_mini.tsv",
    "gpt_legal_rewrite": RUNS_DIR / "bm25_gpt_legal_rewrite_mini.tsv",
    "gpt_keyword_expansion": RUNS_DIR / "bm25_gpt_keyword_expansion_mini.tsv",
    "gpt_hyde_style": RUNS_DIR / "bm25_gpt_hyde_style_mini.tsv",
}


FUSION_CONFIGS = [
    {
        "label": "rrf_original_deepseek_legal_keyword_mini",
        "family": "RRF multi-view",
        "fusion_type": "original_plus_deepseek_legal_keyword",
        "views": ["deepseek_legal_rewrite", "deepseek_keyword_expansion"],
        "include_in_main": True,
    },
    {
        "label": "rrf_original_gpt_legal_keyword_mini",
        "family": "RRF multi-view",
        "fusion_type": "original_plus_gpt_legal_keyword",
        "views": ["gpt_legal_rewrite", "gpt_keyword_expansion"],
        "include_in_main": True,
    },
    {
        "label": "rrf_original_deepseek_keyword_gpt_legal_mini",
        "family": "RRF multi-view",
        "fusion_type": "original_plus_deepseek_keyword_gpt_legal",
        "views": ["deepseek_keyword_expansion", "gpt_legal_rewrite"],
        "include_in_main": True,
    },
    {
        "label": "rrf_original_deepseek_legal_keyword_gpt_legal_mini",
        "family": "RRF multi-view",
        "fusion_type": "original_plus_deepseek_legal_keyword_gpt_legal",
        "views": [
            "deepseek_legal_rewrite",
            "deepseek_keyword_expansion",
            "gpt_legal_rewrite",
        ],
        "include_in_main": True,
    },
    {
        "label": "rrf_original_all_non_hyde_mini",
        "family": "RRF multi-view",
        "fusion_type": "original_plus_all_non_hyde",
        "views": [
            "deepseek_legal_rewrite",
            "deepseek_keyword_expansion",
            "gpt_legal_rewrite",
            "gpt_keyword_expansion",
        ],
        "include_in_main": True,
    },
    {
        "label": "rrf_original_all_views_control_mini",
        "family": "RRF multi-view",
        "fusion_type": "original_plus_all_views_control",
        "views": [
            "deepseek_legal_rewrite",
            "deepseek_keyword_expansion",
            "deepseek_hyde_style",
            "gpt_legal_rewrite",
            "gpt_keyword_expansion",
            "gpt_hyde_style",
        ],
        "include_in_main": False,
    },
    {
        "label": "rcs_original_deepseek_legal_keyword_mini",
        "family": "RCS simple",
        "fusion_type": "rcs_original_plus_deepseek_legal_keyword",
        "views": ["deepseek_legal_rewrite", "deepseek_keyword_expansion"],
        "include_in_main": True,
    },
    {
        "label": "rcs_original_deepseek_keyword_gpt_legal_mini",
        "family": "RCS simple",
        "fusion_type": "rcs_original_plus_deepseek_keyword_gpt_legal",
        "views": ["deepseek_keyword_expansion", "gpt_legal_rewrite"],
        "include_in_main": True,
    },
    {
        "label": "rcs_original_deepseek_legal_keyword_gpt_legal_mini",
        "family": "RCS simple",
        "fusion_type": "rcs_original_plus_deepseek_legal_keyword_gpt_legal",
        "views": [
            "deepseek_legal_rewrite",
            "deepseek_keyword_expansion",
            "gpt_legal_rewrite",
        ],
        "include_in_main": True,
    },
    {
        "label": "rcs_original_all_non_hyde_mini",
        "family": "RCS simple",
        "fusion_type": "rcs_original_plus_all_non_hyde",
        "views": [
            "deepseek_legal_rewrite",
            "deepseek_keyword_expansion",
            "gpt_legal_rewrite",
            "gpt_keyword_expansion",
        ],
        "include_in_main": True,
    },
    {
        "label": "rcs_original_all_views_control_mini",
        "family": "RCS simple",
        "fusion_type": "rcs_original_plus_all_views_control",
        "views": [
            "deepseek_legal_rewrite",
            "deepseek_keyword_expansion",
            "deepseek_hyde_style",
            "gpt_legal_rewrite",
            "gpt_keyword_expansion",
            "gpt_hyde_style",
        ],
        "include_in_main": False,
    },
]


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
        relevant_docs = set(
            group.loc[group["relevance"] > 0, "doc_id"].astype(str)
        )
        qrels[str(qid)] = relevant_docs

    return qrels


def add_rrf_scores(
    scores_by_qid: dict[str, dict[str, float]],
    run_df: pd.DataFrame,
    weight: float,
    rrf_k: int,
) -> None:
    for row in run_df.itertuples(index=False):
        qid = str(row.query_id)
        doc_id = str(row.doc_id)
        rank = int(row.rank)
        scores_by_qid[qid][doc_id] += weight * (1.0 / (rrf_k + rank))


def rrf_fuse(
    *,
    original_df: pd.DataFrame,
    view_dfs: list[pd.DataFrame],
    rrf_k: int,
    top_k: int,
) -> pd.DataFrame:
    scores_by_qid: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    add_rrf_scores(scores_by_qid, original_df, weight=1.0, rrf_k=rrf_k)

    for view_df in view_dfs:
        add_rrf_scores(scores_by_qid, view_df, weight=1.0, rrf_k=rrf_k)

    return scores_to_run(scores_by_qid, top_k=top_k)


def rcs_fuse(
    *,
    original_df: pd.DataFrame,
    view_dfs: list[pd.DataFrame],
    rrf_k: int,
    top_k: int,
    min_votes: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> pd.DataFrame:
    """
    RCS simple :
      - original score pondéré par alpha
      - scores des reformulations pondérés par beta
      - bonus de consensus pondéré par gamma

    Le consensus est calculé uniquement sur les vues de reformulation,
    pas sur l'original.
    """
    scores_by_qid: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    add_rrf_scores(scores_by_qid, original_df, weight=alpha, rrf_k=rrf_k)

    # Ajout des scores reformulation.
    for view_df in view_dfs:
        add_rrf_scores(scores_by_qid, view_df, weight=beta, rrf_k=rrf_k)

    # Consensus reformulation.
    votes_by_qid_doc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    best_rank_by_qid_doc: dict[str, dict[str, int]] = defaultdict(dict)

    for view_df in view_dfs:
        for row in view_df.itertuples(index=False):
            qid = str(row.query_id)
            doc_id = str(row.doc_id)
            rank = int(row.rank)

            votes_by_qid_doc[qid][doc_id] += 1

            if doc_id not in best_rank_by_qid_doc[qid]:
                best_rank_by_qid_doc[qid][doc_id] = rank
            else:
                best_rank_by_qid_doc[qid][doc_id] = min(
                    best_rank_by_qid_doc[qid][doc_id],
                    rank,
                )

    for qid, doc_votes in votes_by_qid_doc.items():
        for doc_id, votes in doc_votes.items():
            if votes < min_votes:
                continue

            best_rank = best_rank_by_qid_doc[qid][doc_id]

            # Bonus comparable à l'échelle RRF.
            consensus_bonus = votes * (1.0 / (rrf_k + best_rank))

            scores_by_qid[qid][doc_id] += gamma * consensus_bonus

    return scores_to_run(scores_by_qid, top_k=top_k)


def scores_to_run(
    scores_by_qid: dict[str, dict[str, float]],
    *,
    top_k: int,
) -> pd.DataFrame:
    rows = []

    for qid, doc_scores in scores_by_qid.items():
        ranked = sorted(
            doc_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]

        for rank, (doc_id, score) in enumerate(ranked, start=1):
            rows.append(
                {
                    "query_id": qid,
                    "doc_id": doc_id,
                    "rank": rank,
                    "score": score,
                }
            )

    return pd.DataFrame(rows)


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
    label: str,
    family: str,
    fusion_type: str,
    views: list[str],
    qrels: dict[str, set[str]],
    query_task: dict[str, str],
    rrf_k: int | None,
    min_votes: int | None,
    alpha: float | None,
    beta: float | None,
    gamma: float | None,
    include_in_main: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_query_df = evaluate_run_per_query(
        run_df=run_df,
        qrels=qrels,
        query_task=query_task,
    )

    per_query_df.insert(0, "method", label)
    per_query_df.insert(1, "family", family)
    per_query_df.insert(2, "fusion_type", fusion_type)
    per_query_df.insert(3, "views", ",".join(views))
    per_query_df.insert(4, "num_views", len(views))
    per_query_df.insert(5, "rrf_k", rrf_k)
    per_query_df.insert(6, "min_votes", min_votes)
    per_query_df.insert(7, "alpha", alpha)
    per_query_df.insert(8, "beta", beta)
    per_query_df.insert(9, "gamma", gamma)
    per_query_df.insert(10, "include_in_main", include_in_main)

    global_df = pd.DataFrame(
        [
            {
                "method": label,
                "dataset": "legalbench_rag_mini",
                "family": family,
                "fusion_type": fusion_type,
                "views": ",".join(views),
                "num_views": len(views),
                "rrf_k": rrf_k,
                "min_votes": min_votes,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "include_in_main": include_in_main,
                **summarize_global(per_query_df),
            }
        ]
    )

    by_task_df = summarize_by_task(per_query_df)
    by_task_df.insert(0, "method", label)
    by_task_df.insert(1, "family", family)
    by_task_df.insert(2, "fusion_type", fusion_type)
    by_task_df.insert(3, "views", ",".join(views))
    by_task_df.insert(4, "num_views", len(views))
    by_task_df.insert(5, "rrf_k", rrf_k)
    by_task_df.insert(6, "min_votes", min_votes)
    by_task_df.insert(7, "alpha", alpha)
    by_task_df.insert(8, "beta", beta)
    by_task_df.insert(9, "gamma", gamma)
    by_task_df.insert(10, "include_in_main", include_in_main)

    return global_df, by_task_df, per_query_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-view RRF and simple RCS on LegalBench-RAG mini."
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
        "--min-votes",
        type=int,
        default=2,
        help="Minimum reformulation votes for RCS consensus bonus.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="RCS weight for original run.",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="RCS weight for reformulation RRF scores.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="RCS weight for consensus bonus.",
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
            f"Missing canonical original run: {ORIGINAL_RUN}"
        )

    print("=" * 80)
    print("LegalBench-RAG mini — multi-view RRF and simple RCS")
    print("=" * 80)

    print(f"Original run: {ORIGINAL_RUN}")
    print(f"top_k = {args.top_k}")
    print(f"rrf_k = {args.rrf_k}")
    print(f"min_votes = {args.min_votes}")
    print(f"alpha = {args.alpha}")
    print(f"beta = {args.beta}")
    print(f"gamma = {args.gamma}")

    queries = read_jsonl(queries_path)
    query_task = {
        str(query["query_id"]): str(query.get("task", "unknown"))
        for query in queries
    }

    qrels_df = load_qrels(qrels_path)
    qrels = build_qrels_dict(qrels_df)

    print("\nLoading runs...")
    original_df = load_run(ORIGINAL_RUN, top_k=args.top_k)

    view_dfs = {}

    for view_name, path in VIEW_RUNS.items():
        print(f"  {view_name}: {path}")
        view_dfs[view_name] = load_run(path, top_k=args.top_k)

    all_global = []
    all_by_task = []
    all_per_query = []

    # Baseline.
    baseline_global, baseline_by_task, baseline_per_query = evaluate_and_format(
        run_df=original_df,
        label="bm25_original_mini_canonical",
        family="Baseline",
        fusion_type="original_only",
        views=[],
        qrels=qrels,
        query_task=query_task,
        rrf_k=None,
        min_votes=None,
        alpha=None,
        beta=None,
        gamma=None,
        include_in_main=True,
    )

    all_global.append(baseline_global)
    all_by_task.append(baseline_by_task)
    all_per_query.append(baseline_per_query)

    print("\nBaseline:")
    print(baseline_global.to_string(index=False))

    config_rows = []

    for config in FUSION_CONFIGS:
        label = config["label"]
        family = config["family"]
        fusion_type = config["fusion_type"]
        views = config["views"]
        include_in_main = bool(config["include_in_main"])

        print("\n" + "=" * 80)
        print(f"Running {label}")
        print("=" * 80)
        print(f"Family: {family}")
        print(f"Views: {views}")

        selected_dfs = [view_dfs[view] for view in views]

        if family == "RRF multi-view":
            fused_df = rrf_fuse(
                original_df=original_df,
                view_dfs=selected_dfs,
                rrf_k=args.rrf_k,
                top_k=args.top_k,
            )

            min_votes = None
            alpha = None
            beta = None
            gamma = None

        elif family == "RCS simple":
            fused_df = rcs_fuse(
                original_df=original_df,
                view_dfs=selected_dfs,
                rrf_k=args.rrf_k,
                top_k=args.top_k,
                min_votes=args.min_votes,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
            )

            min_votes = args.min_votes
            alpha = args.alpha
            beta = args.beta
            gamma = args.gamma

        else:
            raise ValueError(f"Unknown family: {family}")

        fused_df["method"] = label
        fused_df["family"] = family
        fused_df["fusion_type"] = fusion_type
        fused_df["views"] = ",".join(views)

        run_path = RUNS_DIR / f"{label}.tsv"
        fused_df.to_csv(run_path, sep="\t", index=False)

        print(f"Saved run to: {run_path}")

        global_df, by_task_df, per_query_df = evaluate_and_format(
            run_df=fused_df,
            label=label,
            family=family,
            fusion_type=fusion_type,
            views=views,
            qrels=qrels,
            query_task=query_task,
            rrf_k=args.rrf_k,
            min_votes=min_votes,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            include_in_main=include_in_main,
        )

        all_global.append(global_df)
        all_by_task.append(by_task_df)
        all_per_query.append(per_query_df)

        print("\nMetrics:")
        print(global_df.to_string(index=False))

        config_rows.append(
            {
                "method": label,
                "family": family,
                "fusion_type": fusion_type,
                "views": ",".join(views),
                "num_views": len(views),
                "include_in_main": include_in_main,
                "rrf_k": args.rrf_k,
                "min_votes": min_votes,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "run_path": str(run_path),
            }
        )

    metrics_df = pd.concat(all_global, ignore_index=True)
    by_task_df = pd.concat(all_by_task, ignore_index=True)
    per_query_df = pd.concat(all_per_query, ignore_index=True)
    configs_df = pd.DataFrame(config_rows)

    metrics_path = OUT_DIR / "multiview_rrf_rcs_mini_metrics.csv"
    by_task_path = OUT_DIR / "multiview_rrf_rcs_mini_by_task.csv"
    per_query_path = OUT_DIR / "multiview_rrf_rcs_mini_per_query_all.csv"
    configs_path = OUT_DIR / "multiview_rrf_rcs_mini_configs.csv"

    metrics_df.to_csv(metrics_path, index=False)
    by_task_df.to_csv(by_task_path, index=False)
    per_query_df.to_csv(per_query_path, index=False)
    configs_df.to_csv(configs_path, index=False)

    print("\n" + "=" * 80)
    print("All multi-view RRF/RCS metrics")
    print("=" * 80)
    print(metrics_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("All multi-view RRF/RCS metrics by task")
    print("=" * 80)
    print(by_task_df.to_string(index=False))

    print("\nSaved files:")
    print(metrics_path)
    print(by_task_path)
    print(per_query_path)
    print(configs_path)

    print("\nNext:")
    print("  scripts/legalbench/15_multiview_harm_rate_and_stats.py")


if __name__ == "__main__":
    main()