"""Compute harm rates and paired statistics for one-view LegalBench RRF runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("outputs/tables/legalbench")

PER_QUERY_PATH = OUT_DIR / "rrf_original_reformulation_mini_per_query_all.csv"

OUT_GAINS = OUT_DIR / "rrf_original_reformulation_mini_per_query_gains.csv"
OUT_HARM = OUT_DIR / "rrf_original_reformulation_mini_harm_rate.csv"
OUT_HARM_BY_TASK = OUT_DIR / "rrf_original_reformulation_mini_harm_rate_by_task.csv"
OUT_STATS = OUT_DIR / "rrf_original_reformulation_mini_stats.csv"
OUT_SUMMARY = OUT_DIR / "rrf_original_reformulation_mini_summary_for_paper.csv"
OUT_SUMMARY_MD = OUT_DIR / "rrf_original_reformulation_mini_summary_for_paper.md"

METRICS = [
    "Recall@10",
    "Recall@100",
    "MRR@10",
    "nDCG@10",
]

BASELINE_GENERATOR = "none"
BASELINE_QUERY_TYPE = "original"


def classify_gain(gain: float, eps: float) -> str:
    if gain > eps:
        return "improved"

    if gain < -eps:
        return "harmed"

    return "neutral"


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    means = np.empty(n_bootstrap, dtype=float)
    n = len(values)

    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = float(np.mean(sample))

    low = float(np.percentile(means, 100 * alpha / 2))
    high = float(np.percentile(means, 100 * (1 - alpha / 2)))

    return low, high


def sign_flip_p_value(
    values: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
) -> float:
    """
    Approximation Monte Carlo du paired sign-flip test.

    H0 : le signe des gains par query est échangeable.
    Test bilatéral sur la moyenne.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.nan

    observed = abs(float(np.mean(values)))

    if observed == 0:
        return 1.0

    if np.allclose(values, 0.0):
        return 1.0

    rng = np.random.default_rng(seed)
    n = len(values)

    count = 0

    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=n, replace=True)
        permuted_mean = abs(float(np.mean(values * signs)))

        if permuted_mean >= observed:
            count += 1

    # +1 smoothing pour éviter p=0.
    return float((count + 1) / (n_permutations + 1))


def holm_correction(p_values: list[float]) -> list[float]:
    """
    Correction Holm-Bonferroni.
    Retourne les p-values ajustées dans l'ordre original.
    """
    m = len(p_values)

    if m == 0:
        return []

    indexed = sorted(
        enumerate(p_values),
        key=lambda item: float("inf") if pd.isna(item[1]) else item[1],
    )

    adjusted = [np.nan] * m
    running_max = 0.0

    for rank, (original_index, p_value) in enumerate(indexed, start=1):
        if pd.isna(p_value):
            adjusted[original_index] = np.nan
            continue

        raw_adjusted = (m - rank + 1) * p_value
        running_max = max(running_max, raw_adjusted)
        adjusted[original_index] = min(1.0, running_max)

    return adjusted


def load_per_query() -> pd.DataFrame:
    if not PER_QUERY_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {PER_QUERY_PATH}\n"
            "Run script 12_rrf_original_reformulation.py first."
        )

    df = pd.read_csv(PER_QUERY_PATH, dtype={"query_id": str})

    required_cols = {
        "method",
        "generator",
        "query_type",
        "fusion",
        "query_id",
        "task",
        *METRICS,
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns in {PER_QUERY_PATH}: {missing}")

    return df


def build_gain_table(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    baseline = df[
        (df["generator"] == BASELINE_GENERATOR)
        & (df["query_type"] == BASELINE_QUERY_TYPE)
    ].copy()

    if baseline.empty:
        raise ValueError("Baseline rows not found in per-query RRF table.")

    baseline = baseline[["query_id", "task", *METRICS]].copy()

    baseline = baseline.rename(
        columns={metric: f"baseline_{metric}" for metric in METRICS}
    )

    candidates = df[
        ~(
            (df["generator"] == BASELINE_GENERATOR)
            & (df["query_type"] == BASELINE_QUERY_TYPE)
        )
    ].copy()

    merged = candidates.merge(
        baseline,
        on=["query_id", "task"],
        how="left",
        validate="many_to_one",
    )

    for metric in METRICS:
        merged[f"gain_{metric}"] = (
            merged[metric] - merged[f"baseline_{metric}"]
        )

        merged[f"status_{metric}"] = merged[f"gain_{metric}"].apply(
            lambda x: classify_gain(float(x), eps)
        )

    return merged


def make_harm_summary(gains_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = [
        "method",
        "generator",
        "query_type",
        "fusion",
        "rrf_k",
    ]

    for group_key, group in gains_df.groupby(group_cols, dropna=False):
        group_dict = dict(zip(group_cols, group_key))

        for metric in METRICS:
            baseline_metric = f"baseline_{metric}"
            gain_metric = f"gain_{metric}"
            status_metric = f"status_{metric}"

            num_queries = len(group)
            num_improved = int((group[status_metric] == "improved").sum())
            num_harmed = int((group[status_metric] == "harmed").sum())
            num_neutral = int((group[status_metric] == "neutral").sum())

            rows.append(
                {
                    **group_dict,
                    "metric": metric,
                    "baseline": float(group[baseline_metric].mean()),
                    "fusion_score": float(group[metric].mean()),
                    "delta": float(group[gain_metric].mean()),
                    "num_queries": int(num_queries),
                    "num_improved": num_improved,
                    "num_harmed": num_harmed,
                    "num_neutral": num_neutral,
                    "improve_%": 100.0 * num_improved / num_queries,
                    "harm_%": 100.0 * num_harmed / num_queries,
                    "neutral_%": 100.0 * num_neutral / num_queries,
                    "improved/harmed/neutral": (
                        f"{num_improved}/{num_harmed}/{num_neutral}"
                    ),
                }
            )

    return pd.DataFrame(rows)


def make_harm_summary_by_task(gains_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = [
        "method",
        "generator",
        "query_type",
        "fusion",
        "rrf_k",
        "task",
    ]

    for group_key, group in gains_df.groupby(group_cols, dropna=False):
        group_dict = dict(zip(group_cols, group_key))

        for metric in METRICS:
            baseline_metric = f"baseline_{metric}"
            gain_metric = f"gain_{metric}"
            status_metric = f"status_{metric}"

            num_queries = len(group)
            num_improved = int((group[status_metric] == "improved").sum())
            num_harmed = int((group[status_metric] == "harmed").sum())
            num_neutral = int((group[status_metric] == "neutral").sum())

            rows.append(
                {
                    **group_dict,
                    "metric": metric,
                    "baseline": float(group[baseline_metric].mean()),
                    "fusion_score": float(group[metric].mean()),
                    "delta": float(group[gain_metric].mean()),
                    "num_queries": int(num_queries),
                    "num_improved": num_improved,
                    "num_harmed": num_harmed,
                    "num_neutral": num_neutral,
                    "improve_%": 100.0 * num_improved / num_queries,
                    "harm_%": 100.0 * num_harmed / num_queries,
                    "neutral_%": 100.0 * num_neutral / num_queries,
                    "improved/harmed/neutral": (
                        f"{num_improved}/{num_harmed}/{num_neutral}"
                    ),
                }
            )

    return pd.DataFrame(rows)


def make_stats(
    gains_df: pd.DataFrame,
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rows = []

    group_cols = [
        "method",
        "generator",
        "query_type",
        "fusion",
        "rrf_k",
    ]

    counter = 0

    for group_key, group in gains_df.groupby(group_cols, dropna=False):
        group_dict = dict(zip(group_cols, group_key))

        for metric in METRICS:
            gain_metric = f"gain_{metric}"
            values = group[gain_metric].to_numpy(dtype=float)

            ci_low, ci_high = bootstrap_ci(
                values,
                n_bootstrap=n_bootstrap,
                seed=seed + counter,
            )

            p_raw = sign_flip_p_value(
                values,
                n_permutations=n_permutations,
                seed=seed + 10000 + counter,
            )

            mean_gain = float(np.mean(values))

            rows.append(
                {
                    **group_dict,
                    "metric": metric,
                    "mean_gain": mean_gain,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "p_raw_sign_flip": p_raw,
                    "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
                }
            )

            counter += 1

    stats_df = pd.DataFrame(rows)

    stats_df["p_holm"] = holm_correction(
        stats_df["p_raw_sign_flip"].tolist()
    )

    stats_df["significant_holm_0.05"] = stats_df["p_holm"] < 0.05

    # Réordonner les colonnes.
    ordered_cols = [
        "method",
        "generator",
        "query_type",
        "fusion",
        "rrf_k",
        "metric",
        "mean_gain",
        "ci95_low",
        "ci95_high",
        "p_raw_sign_flip",
        "p_holm",
        "ci_excludes_zero",
        "significant_holm_0.05",
    ]

    return stats_df[ordered_cols]


def make_summary_for_paper(
    harm_df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = harm_df.merge(
        stats_df[
            [
                "method",
                "generator",
                "query_type",
                "fusion",
                "rrf_k",
                "metric",
                "ci95_low",
                "ci95_high",
                "p_raw_sign_flip",
                "p_holm",
                "ci_excludes_zero",
                "significant_holm_0.05",
            ]
        ],
        on=["method", "generator", "query_type", "fusion", "rrf_k", "metric"],
        how="left",
    )

    # Colonnes paper-friendly.
    merged["delta"] = merged["delta"].map(lambda x: f"{x:+.4f}")
    merged["baseline"] = merged["baseline"].map(lambda x: f"{x:.6f}")
    merged["fusion_score"] = merged["fusion_score"].map(lambda x: f"{x:.6f}")
    merged["harm_%"] = merged["harm_%"].map(lambda x: f"{x:.2f}")
    merged["improve_%"] = merged["improve_%"].map(lambda x: f"{x:.2f}")
    merged["neutral_%"] = merged["neutral_%"].map(lambda x: f"{x:.2f}")
    merged["ci95_low"] = merged["ci95_low"].map(lambda x: f"{x:+.4f}")
    merged["ci95_high"] = merged["ci95_high"].map(lambda x: f"{x:+.4f}")
    merged["p_holm"] = merged["p_holm"].map(lambda x: f"{x:.4f}")

    return merged


def write_markdown(summary_df: pd.DataFrame) -> None:
    lines = []

    lines.append("# LegalBench-RAG mini — RRF original + reformulation")
    lines.append("")
    lines.append("This table compares each RRF fusion against the canonical BM25 original baseline.")
    lines.append("")
    lines.append(summary_df.to_markdown(index=False))
    lines.append("")

    OUT_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute harm-rate and statistics for LegalBench RRF results."
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Tolerance for classifying neutral gains.",
    )

    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples for CI.",
    )

    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of sign-flip permutations.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("LegalBench-RAG mini — RRF harm-rate and statistics")
    print("=" * 80)

    print(f"Input per-query file: {PER_QUERY_PATH}")
    print(f"eps = {args.eps}")
    print(f"n_bootstrap = {args.n_bootstrap}")
    print(f"n_permutations = {args.n_permutations}")
    print(f"seed = {args.seed}")

    df = load_per_query()

    print(f"Rows loaded: {len(df)}")
    print(f"Methods: {df['method'].nunique()}")
    print(f"Queries: {df['query_id'].nunique()}")

    print("\nBuilding gain table...")
    gains_df = build_gain_table(df, eps=args.eps)

    print("Computing harm-rate summary...")
    harm_df = make_harm_summary(gains_df)

    print("Computing harm-rate summary by task...")
    harm_by_task_df = make_harm_summary_by_task(gains_df)

    print("Computing bootstrap CIs and sign-flip p-values...")
    stats_df = make_stats(
        gains_df,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )

    print("Building paper summary...")
    summary_df = make_summary_for_paper(harm_df, stats_df)

    gains_df.to_csv(OUT_GAINS, index=False)
    harm_df.to_csv(OUT_HARM, index=False)
    harm_by_task_df.to_csv(OUT_HARM_BY_TASK, index=False)
    stats_df.to_csv(OUT_STATS, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    write_markdown(summary_df)

    print("\n" + "=" * 80)
    print("RRF harm-rate summary")
    print("=" * 80)
    print(harm_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("RRF statistical summary")
    print("=" * 80)
    print(stats_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("RRF summary for paper")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print("\nSaved files:")
    print(OUT_GAINS)
    print(OUT_HARM)
    print(OUT_HARM_BY_TASK)
    print(OUT_STATS)
    print(OUT_SUMMARY)
    print(OUT_SUMMARY_MD)

    print("\nNext:")
    print("  Interpret which RRF configurations are robust.")
    print("  Then build multi-view RRF / RCS if needed.")


if __name__ == "__main__":
    main()