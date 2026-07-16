"""Evaluate harm and paired significance for multi-view BSARD RRF runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RUNS_DIR = Path("runs/bsard")
TABLE_DIR = Path("outputs/tables/bsard")
QRELS_PATH = Path("data/raw/bsard/qrels_test.tsv")

BASELINE_PER_QUERY_PATH = TABLE_DIR / "bm25_original_test_per_query.csv"

FUSION_RUNS = [
    {
        "method": "rrf_bm25_original_deepseek_all_reformulations_test",
        "generator": "deepseek",
        "fusion_type": "original_plus_all_deepseek",
        "run_path": RUNS_DIR / "rrf_bm25_original_deepseek_all_reformulations_test.tsv",
    },
    {
        "method": "rrf_bm25_original_gpt_all_reformulations_test",
        "generator": "gpt",
        "fusion_type": "original_plus_all_gpt",
        "run_path": RUNS_DIR / "rrf_bm25_original_gpt_all_reformulations_test.tsv",
    },
    {
        "method": "rrf_bm25_original_all_generators_all_reformulations_test",
        "generator": "deepseek+gpt",
        "fusion_type": "original_plus_all_generators",
        "run_path": RUNS_DIR / "rrf_bm25_original_all_generators_all_reformulations_test.tsv",
    },
]

METRICS = ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]

N_BOOTSTRAP = 10000
N_RANDOMIZATION = 10000
RANDOM_SEED = 42
EPS = 1e-12


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


def load_run(path: Path) -> pd.DataFrame:
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
        raise ValueError(f"Missing columns in {path}: {missing}")

    df["rank"] = df["rank"].astype(int)

    return df


def compute_query_metrics(
    qid: str,
    ranked_docs: list[str],
    relevant_docs: set[str],
) -> dict:
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


def evaluate_run_per_query(run_df: pd.DataFrame, qrels: dict[str, set[str]]) -> pd.DataFrame:
    grouped_run = run_df.groupby("query_id", sort=False)

    rows = []

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

        rows.append(
            compute_query_metrics(
                qid=qid,
                ranked_docs=ranked_docs,
                relevant_docs=relevant_docs,
            )
        )

    return pd.DataFrame(rows)


def classify_gain(gain: float) -> str:
    if gain > EPS:
        return "improved"
    if gain < -EPS:
        return "harmed"
    return "neutral"


def bootstrap_ci_mean(
    values: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)

    boot_means = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return float(ci_low), float(ci_high)


def sign_flip_p_value(
    gains: np.ndarray,
    n_randomization: int = N_RANDOMIZATION,
    seed: int = RANDOM_SEED,
) -> float:
    rng = np.random.default_rng(seed)

    observed = abs(float(gains.mean()))
    n = len(gains)

    count = 0

    for _ in range(n_randomization):
        signs = rng.choice([-1.0, 1.0], size=n, replace=True)
        randomized_mean = abs(float((gains * signs).mean()))

        if randomized_mean >= observed - EPS:
            count += 1

    return float((count + 1) / (n_randomization + 1))


def holm_bonferroni(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)

    adjusted = np.empty(m, dtype=float)
    running_max = 0.0

    for rank, idx in enumerate(order, start=1):
        raw_p = p_values[idx]
        adj_p = (m - rank + 1) * raw_p
        running_max = max(running_max, adj_p)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted.tolist()


def format_signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def main() -> None:
    if not BASELINE_PER_QUERY_PATH.exists():
        raise FileNotFoundError(
            f"Missing baseline per-query file: {BASELINE_PER_QUERY_PATH}"
        )

    print("Loading qrels...")
    qrels = build_qrels_dict(load_qrels(QRELS_PATH))

    print("Loading baseline per-query metrics...")
    baseline = pd.read_csv(
        BASELINE_PER_QUERY_PATH,
        dtype={"query_id": str},
    )

    baseline = baseline[["query_id", *METRICS]].copy()
    baseline = baseline.rename(columns={m: f"{m}_baseline" for m in METRICS})

    all_gain_rows = []
    harm_rows = []
    stats_rows = []
    paper_rows = []

    for fusion in FUSION_RUNS:
        method = fusion["method"]
        generator = fusion["generator"]
        fusion_type = fusion["fusion_type"]
        run_path = fusion["run_path"]

        print("\n" + "=" * 80)
        print(f"Evaluating multi-view fusion: {method}")
        print(f"Run file: {run_path}")

        run_df = load_run(run_path)
        per_query = evaluate_run_per_query(run_df, qrels)

        per_query["method"] = method
        per_query["generator"] = generator
        per_query["fusion_type"] = fusion_type

        candidate = per_query[["query_id", *METRICS]].copy()
        candidate = candidate.rename(columns={m: f"{m}_candidate" for m in METRICS})

        merged = baseline.merge(candidate, on="query_id", how="inner")

        if len(merged) != len(baseline):
            raise ValueError(
                f"Query mismatch for {method}: "
                f"{len(merged)} merged vs {len(baseline)} baseline."
            )

        merged.insert(1, "method", method)
        merged.insert(2, "generator", generator)
        merged.insert(3, "fusion_type", fusion_type)

        for metric in METRICS:
            base_col = f"{metric}_baseline"
            cand_col = f"{metric}_candidate"
            gain_col = f"{metric}_gain"
            status_col = f"{metric}_status"

            merged[gain_col] = merged[cand_col] - merged[base_col]
            merged[status_col] = merged[gain_col].apply(classify_gain)

            gains = merged[gain_col].to_numpy(dtype=float)

            num_queries = len(gains)
            num_improved = int((gains > EPS).sum())
            num_harmed = int((gains < -EPS).sum())
            num_neutral = int((np.abs(gains) <= EPS).sum())

            mean_baseline = float(merged[base_col].mean())
            mean_candidate = float(merged[cand_col].mean())
            mean_gain = float(gains.mean())

            ci_low, ci_high = bootstrap_ci_mean(gains)
            p_raw = sign_flip_p_value(gains)

            harm_rows.append(
                {
                    "method": method,
                    "generator": generator,
                    "fusion_type": fusion_type,
                    "metric": metric,
                    "mean_baseline": mean_baseline,
                    "mean_candidate": mean_candidate,
                    "mean_gain": mean_gain,
                    "num_queries": num_queries,
                    "num_improved": num_improved,
                    "num_harmed": num_harmed,
                    "num_neutral": num_neutral,
                    "improve_rate": num_improved / num_queries,
                    "harm_rate": num_harmed / num_queries,
                    "neutral_rate": num_neutral / num_queries,
                }
            )

            stats_rows.append(
                {
                    "method": method,
                    "generator": generator,
                    "fusion_type": fusion_type,
                    "metric": metric,
                    "num_queries": num_queries,
                    "mean_baseline": mean_baseline,
                    "mean_candidate": mean_candidate,
                    "mean_gain": mean_gain,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "p_raw_sign_flip": p_raw,
                    "num_positive": num_improved,
                    "num_negative": num_harmed,
                    "num_zero": num_neutral,
                }
            )

            paper_rows.append(
                {
                    "generator": generator,
                    "fusion_type": fusion_type,
                    "metric": metric,
                    "baseline": round(mean_baseline, 6),
                    "fusion_score": round(mean_candidate, 6),
                    "delta": format_signed(mean_gain),
                    "harm_%": round((num_harmed / num_queries) * 100, 2),
                    "improve_%": round((num_improved / num_queries) * 100, 2),
                    "neutral_%": round((num_neutral / num_queries) * 100, 2),
                    "improved/harmed/neutral": (
                        f"{num_improved}/{num_harmed}/{num_neutral}"
                    ),
                }
            )

        all_gain_rows.append(merged)

    gains_df = pd.concat(all_gain_rows, ignore_index=True)
    harm_df = pd.DataFrame(harm_rows)
    stats_df = pd.DataFrame(stats_rows)
    paper_df = pd.DataFrame(paper_rows)

    stats_df["p_holm"] = holm_bonferroni(
        stats_df["p_raw_sign_flip"].tolist()
    )

    stats_df["ci_excludes_zero"] = (
        (stats_df["ci95_low"] > 0) | (stats_df["ci95_high"] < 0)
    )

    stats_df["significant_raw_0.05"] = stats_df["p_raw_sign_flip"] < 0.05
    stats_df["significant_holm_0.05"] = stats_df["p_holm"] < 0.05

    fusion_order = {
        "original_plus_all_deepseek": 1,
        "original_plus_all_gpt": 2,
        "original_plus_all_generators": 3,
    }

    metric_order = {
        "Recall@10": 1,
        "Recall@100": 2,
        "MRR@10": 3,
        "nDCG@10": 4,
    }

    for df in [harm_df, stats_df, paper_df]:
        df["fusion_order"] = df["fusion_type"].map(fusion_order)
        df["metric_order"] = df["metric"].map(metric_order)
        df.sort_values(["fusion_order", "metric_order"], inplace=True)
        df.drop(columns=["fusion_order", "metric_order"], inplace=True)

    gains_path = TABLE_DIR / "rrf_all_reformulations_per_query_all.csv"
    harm_path = TABLE_DIR / "rrf_all_reformulations_harm_rate.csv"
    stats_path = TABLE_DIR / "rrf_all_reformulations_stats.csv"
    paper_path = TABLE_DIR / "rrf_all_reformulations_summary_for_paper.csv"

    gains_df.to_csv(gains_path, index=False)
    harm_df.to_csv(harm_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    paper_df.to_csv(paper_path, index=False)

    print("\n" + "=" * 80)
    print("Multi-view fusion summary for paper:")
    print(paper_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Multi-view fusion statistical summary:")
    display_cols = [
        "generator",
        "fusion_type",
        "metric",
        "mean_gain",
        "ci95_low",
        "ci95_high",
        "p_raw_sign_flip",
        "p_holm",
        "ci_excludes_zero",
        "significant_holm_0.05",
    ]
    print(stats_df[display_cols].to_string(index=False))

    print("\nSaved files:")
    print(gains_path)
    print(harm_path)
    print(stats_path)
    print(paper_path)


if __name__ == "__main__":
    main()