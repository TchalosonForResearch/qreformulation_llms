"""Evaluate harm and significance for LegalBench multi-view runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("outputs/tables/legalbench")

PER_QUERY_PATH = OUT_DIR / "multiview_rrf_rcs_mini_per_query_all.csv"

OUT_GAINS = OUT_DIR / "multiview_rrf_rcs_mini_per_query_gains.csv"
OUT_HARM = OUT_DIR / "multiview_rrf_rcs_mini_harm_rate.csv"
OUT_HARM_BY_TASK = OUT_DIR / "multiview_rrf_rcs_mini_harm_rate_by_task.csv"
OUT_STATS = OUT_DIR / "multiview_rrf_rcs_mini_stats.csv"
OUT_SUMMARY = OUT_DIR / "multiview_rrf_rcs_mini_summary_for_paper.csv"
OUT_SUMMARY_MD = OUT_DIR / "multiview_rrf_rcs_mini_summary_for_paper.md"

BASELINE_METHOD = "bm25_original_mini_canonical"

METRICS = [
    "Recall@10",
    "Recall@100",
    "MRR@10",
    "nDCG@10",
]


META_COLS = [
    "method",
    "family",
    "fusion_type",
    "views",
    "num_views",
    "rrf_k",
    "min_votes",
    "alpha",
    "beta",
    "gamma",
    "include_in_main",
]


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
    n = len(values)

    means = np.empty(n_bootstrap, dtype=float)

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
    Paired sign-flip test, Monte Carlo approximation.

    H0:
      The signs of per-query gains are exchangeable.

    Two-sided test on the mean gain.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.nan

    if np.allclose(values, 0.0):
        return 1.0

    observed = abs(float(np.mean(values)))

    if observed == 0:
        return 1.0

    rng = np.random.default_rng(seed)
    n = len(values)

    count = 0

    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=n, replace=True)
        permuted_mean = abs(float(np.mean(values * signs)))

        if permuted_mean >= observed:
            count += 1

    return float((count + 1) / (n_permutations + 1))


def holm_correction(p_values: list[float]) -> list[float]:
    """
    Holm-Bonferroni correction.
    Returns adjusted p-values in the original order.
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
            "Run script 14_rrf_multiview_and_rcs_simple.py first."
        )

    df = pd.read_csv(PER_QUERY_PATH, dtype={"query_id": str})

    required = {
        "method",
        "family",
        "fusion_type",
        "query_id",
        "task",
        *METRICS,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns in {PER_QUERY_PATH}: {missing}")

    for col in META_COLS:
        if col not in df.columns:
            df[col] = np.nan

    return df


def build_gain_table(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    baseline = df[df["method"] == BASELINE_METHOD].copy()

    if baseline.empty:
        raise ValueError(f"Baseline method not found: {BASELINE_METHOD}")

    baseline = baseline[["query_id", "task", *METRICS]].copy()

    baseline = baseline.rename(
        columns={metric: f"baseline_{metric}" for metric in METRICS}
    )

    candidates = df[df["method"] != BASELINE_METHOD].copy()

    merged = candidates.merge(
        baseline,
        on=["query_id", "task"],
        how="left",
        validate="many_to_one",
    )

    if merged[[f"baseline_{metric}" for metric in METRICS]].isna().any().any():
        raise ValueError("Some candidate rows could not be matched to baseline rows.")

    for metric in METRICS:
        gain_col = f"gain_{metric}"
        status_col = f"status_{metric}"

        merged[gain_col] = merged[metric] - merged[f"baseline_{metric}"]

        merged[status_col] = merged[gain_col].apply(
            lambda x: classify_gain(float(x), eps)
        )

    return merged


def make_harm_summary(gains_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for method, group in gains_df.groupby("method", dropna=False):
        meta = group.iloc[0][META_COLS].to_dict()

        for metric in METRICS:
            baseline_col = f"baseline_{metric}"
            gain_col = f"gain_{metric}"
            status_col = f"status_{metric}"

            num_queries = int(len(group))
            num_improved = int((group[status_col] == "improved").sum())
            num_harmed = int((group[status_col] == "harmed").sum())
            num_neutral = int((group[status_col] == "neutral").sum())

            rows.append(
                {
                    **meta,
                    "metric": metric,
                    "baseline": float(group[baseline_col].mean()),
                    "candidate": float(group[metric].mean()),
                    "delta": float(group[gain_col].mean()),
                    "num_queries": num_queries,
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

    for (method, task), group in gains_df.groupby(["method", "task"], dropna=False):
        meta = group.iloc[0][META_COLS].to_dict()

        for metric in METRICS:
            baseline_col = f"baseline_{metric}"
            gain_col = f"gain_{metric}"
            status_col = f"status_{metric}"

            num_queries = int(len(group))
            num_improved = int((group[status_col] == "improved").sum())
            num_harmed = int((group[status_col] == "harmed").sum())
            num_neutral = int((group[status_col] == "neutral").sum())

            rows.append(
                {
                    **meta,
                    "task": task,
                    "metric": metric,
                    "baseline": float(group[baseline_col].mean()),
                    "candidate": float(group[metric].mean()),
                    "delta": float(group[gain_col].mean()),
                    "num_queries": num_queries,
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
    counter = 0

    for method, group in gains_df.groupby("method", dropna=False):
        meta = group.iloc[0][META_COLS].to_dict()

        for metric in METRICS:
            gain_col = f"gain_{metric}"
            values = group[gain_col].to_numpy(dtype=float)

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

            rows.append(
                {
                    **meta,
                    "metric": metric,
                    "mean_gain": float(np.mean(values)),
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

    ordered_cols = [
        *META_COLS,
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
    stats_keep = [
        "method",
        "metric",
        "ci95_low",
        "ci95_high",
        "p_raw_sign_flip",
        "p_holm",
        "ci_excludes_zero",
        "significant_holm_0.05",
    ]

    merged = harm_df.merge(
        stats_df[stats_keep],
        on=["method", "metric"],
        how="left",
        validate="one_to_one",
    )

    # Human-friendly ordering.
    family_order = {
        "RCS simple": 0,
        "RRF multi-view": 1,
    }

    metric_order = {
        "Recall@10": 0,
        "Recall@100": 1,
        "MRR@10": 2,
        "nDCG@10": 3,
    }

    merged["_family_order"] = merged["family"].map(family_order).fillna(99)
    merged["_metric_order"] = merged["metric"].map(metric_order).fillna(99)

    merged = merged.sort_values(
        ["include_in_main", "_family_order", "method", "_metric_order"],
        ascending=[False, True, True, True],
    ).drop(columns=["_family_order", "_metric_order"])

    return merged


def make_paper_friendly(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()

    df["baseline"] = df["baseline"].map(lambda x: f"{x:.6f}")
    df["candidate"] = df["candidate"].map(lambda x: f"{x:.6f}")
    df["delta"] = df["delta"].map(lambda x: f"{x:+.4f}")
    df["harm_%"] = df["harm_%"].map(lambda x: f"{x:.2f}")
    df["improve_%"] = df["improve_%"].map(lambda x: f"{x:.2f}")
    df["neutral_%"] = df["neutral_%"].map(lambda x: f"{x:.2f}")
    df["ci95_low"] = df["ci95_low"].map(lambda x: f"{x:+.4f}")
    df["ci95_high"] = df["ci95_high"].map(lambda x: f"{x:+.4f}")
    df["p_raw_sign_flip"] = df["p_raw_sign_flip"].map(lambda x: f"{x:.6f}")
    df["p_holm"] = df["p_holm"].map(lambda x: f"{x:.4f}")

    return df


def write_markdown(summary_df: pd.DataFrame) -> None:
    main_df = summary_df[summary_df["include_in_main"] == True].copy()
    control_df = summary_df[summary_df["include_in_main"] == False].copy()

    friendly_main = make_paper_friendly(main_df)
    friendly_control = make_paper_friendly(control_df)

    lines = []

    lines.append("# LegalBench-RAG mini — Multi-view RRF/RCS harm-rate and statistics")
    lines.append("")
    lines.append(
        "All rows compare the candidate fusion against the canonical BM25 original baseline."
    )
    lines.append("")
    lines.append("## Main configurations")
    lines.append("")
    lines.append("```text")
    lines.append(friendly_main.to_string(index=False))
    lines.append("```")
    lines.append("")

    if not friendly_control.empty:
        lines.append("## Control configurations")
        lines.append("")
        lines.append("```text")
        lines.append(friendly_control.to_string(index=False))
        lines.append("```")
        lines.append("")

    OUT_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute harm-rate and statistics for LegalBench multi-view RRF/RCS."
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Tolerance for neutral gain classification.",
    )

    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples.",
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
    print("LegalBench-RAG mini — multi-view RRF/RCS harm-rate and statistics")
    print("=" * 80)

    print(f"Input per-query file: {PER_QUERY_PATH}")
    print(f"eps = {args.eps}")
    print(f"n_bootstrap = {args.n_bootstrap}")
    print(f"n_permutations = {args.n_permutations}")
    print(f"seed = {args.seed}")

    df = load_per_query()

    print(f"\nRows loaded: {len(df)}")
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
    print("Multi-view harm-rate summary")
    print("=" * 80)
    print(harm_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Multi-view statistical summary")
    print("=" * 80)
    print(stats_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Multi-view summary for paper")
    print("=" * 80)
    print(make_paper_friendly(summary_df).to_string(index=False))

    print("\nSaved files:")
    print(OUT_GAINS)
    print(OUT_HARM)
    print(OUT_HARM_BY_TASK)
    print(OUT_STATS)
    print(OUT_SUMMARY)
    print(OUT_SUMMARY_MD)

    print("\nNext:")
    print("  Select the LegalBench final synthesis rows.")
    print("  Then build the combined BSARD + LegalBench final table.")


if __name__ == "__main__":
    main()