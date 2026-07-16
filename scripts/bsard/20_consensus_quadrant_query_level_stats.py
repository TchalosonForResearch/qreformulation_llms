"""Compute query-level statistics for BSARD consensus quadrants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TABLE_DIR = Path("outputs/tables/bsard")

INPUT_PATH = TABLE_DIR / "consensus_quadrant_per_doc.csv"

QUERY_LEVEL_RATES_PATH = TABLE_DIR / "consensus_quadrant_query_level_rates.csv"
SUMMARY_PATH = TABLE_DIR / "consensus_quadrant_query_level_summary.csv"
PAIRWISE_STATS_PATH = TABLE_DIR / "consensus_quadrant_query_level_pairwise_stats.csv"
PAIRWISE_MD_PATH = TABLE_DIR / "consensus_quadrant_query_level_pairwise_stats.md"

N_BOOTSTRAP = 10000
N_RANDOMIZATION = 10000
RANDOM_SEED = 42
EPS = 1e-12

GROUP_ORDER = {
    "deepseek_only": 1,
    "gpt_only": 2,
    "all_generators": 3,
}

QUADRANT_ORDER = {
    "anchor_high__consensus_high": 1,
    "anchor_high__consensus_low": 2,
    "anchor_low__consensus_high": 3,
    "anchor_low__consensus_low": 4,
}

REFERENCE_QUADRANT = "anchor_high__consensus_high"

COMPARISON_QUADRANTS = [
    "anchor_high__consensus_low",
    "anchor_low__consensus_high",
    "anchor_low__consensus_low",
]


def bootstrap_ci_mean(
    values: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """
    IC bootstrap percentile à 95 % pour une moyenne.
    """
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = len(values)

    boot_means = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return float(ci_low), float(ci_high)


def sign_flip_p_value(
    diffs: np.ndarray,
    n_randomization: int = N_RANDOMIZATION,
    seed: int = RANDOM_SEED,
) -> float:
    """
    Sign-flip test bilatéral sur des différences appariées.
    """
    diffs = diffs[~np.isnan(diffs)]

    if len(diffs) == 0:
        return np.nan

    rng = np.random.default_rng(seed)

    observed = abs(float(diffs.mean()))
    n = len(diffs)

    count = 0

    for _ in range(n_randomization):
        signs = rng.choice([-1.0, 1.0], size=n, replace=True)
        randomized_mean = abs(float((diffs * signs).mean()))

        if randomized_mean >= observed - EPS:
            count += 1

    return float((count + 1) / (n_randomization + 1))


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """
    Correction Holm-Bonferroni.
    Ignore les NaN puis les remet à NaN.
    """
    p_values_array = np.array(p_values, dtype=float)

    valid_mask = ~np.isnan(p_values_array)
    valid_indices = np.where(valid_mask)[0]
    valid_p = p_values_array[valid_mask]

    adjusted = np.full(len(p_values), np.nan, dtype=float)

    if len(valid_p) == 0:
        return adjusted.tolist()

    m = len(valid_p)
    order = np.argsort(valid_p)

    running_max = 0.0

    for rank, local_idx in enumerate(order, start=1):
        raw_p = valid_p[local_idx]
        adj_p = (m - rank + 1) * raw_p
        running_max = max(running_max, adj_p)

        original_idx = valid_indices[local_idx]
        adjusted[original_idx] = min(running_max, 1.0)

    return adjusted.tolist()


def make_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []

    for _, row in df.iterrows():
        values = [str(row[col]) for col in columns]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *rows])


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}\n"
            "Lance d'abord scripts/bsard/15_consensus_quadrant_analysis.py"
        )

    print(f"Loading per-doc consensus data: {INPUT_PATH}")
    df = pd.read_csv(
        INPUT_PATH,
        dtype={"query_id": str, "doc_id": str},
    )

    required = {"group", "query_id", "doc_id", "quadrant", "is_relevant"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in {INPUT_PATH}: {missing}")

    # Convertit is_relevant en bool si besoin.
    if df["is_relevant"].dtype != bool:
        df["is_relevant"] = df["is_relevant"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )

    print("Computing query-level quadrant relevance rates...")

    query_rows = []

    grouped = df.groupby(["group", "query_id", "quadrant"], sort=False)

    for (group, query_id, quadrant), chunk in grouped:
        num_pairs = len(chunk)
        num_relevant = int(chunk["is_relevant"].sum())
        relevance_rate = num_relevant / num_pairs if num_pairs > 0 else np.nan

        query_rows.append(
            {
                "group": group,
                "query_id": query_id,
                "quadrant": quadrant,
                "num_pairs": num_pairs,
                "num_relevant": num_relevant,
                "relevance_rate": relevance_rate,
            }
        )

    query_df = pd.DataFrame(query_rows)
    query_df.to_csv(QUERY_LEVEL_RATES_PATH, index=False)

    print(f"Saved query-level rates to: {QUERY_LEVEL_RATES_PATH}")

    # Résumé par group/quadrant.
    summary_rows = []

    for (group, quadrant), chunk in query_df.groupby(["group", "quadrant"], sort=False):
        rates = chunk["relevance_rate"].to_numpy(dtype=float)

        mean_rate = float(np.nanmean(rates))
        median_rate = float(np.nanmedian(rates))
        ci_low, ci_high = bootstrap_ci_mean(rates)

        summary_rows.append(
            {
                "group": group,
                "quadrant": quadrant,
                "num_queries_present": int(chunk["query_id"].nunique()),
                "mean_query_relevance_rate": mean_rate,
                "median_query_relevance_rate": median_rate,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "mean_query_relevance_%": round(mean_rate * 100, 4),
                "ci95_low_%": round(ci_low * 100, 4),
                "ci95_high_%": round(ci_high * 100, 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_df["group_order"] = summary_df["group"].map(GROUP_ORDER)
    summary_df["quadrant_order"] = summary_df["quadrant"].map(QUADRANT_ORDER)

    summary_df = summary_df.sort_values(
        ["group_order", "quadrant_order"]
    ).drop(columns=["group_order", "quadrant_order"])

    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\n" + "=" * 80)
    print("Query-level quadrant summary:")
    print(summary_df.to_string(index=False))

    # Comparaisons appariées :
    # anchor_high__consensus_high vs les trois autres quadrants.
    print("\nComputing paired query-level comparisons...")

    pairwise_rows = []

    for group in sorted(query_df["group"].unique(), key=lambda x: GROUP_ORDER.get(x, 999)):
        group_df = query_df[query_df["group"] == group].copy()

        pivot = group_df.pivot_table(
            index="query_id",
            columns="quadrant",
            values="relevance_rate",
            aggfunc="mean",
        )

        if REFERENCE_QUADRANT not in pivot.columns:
            continue

        for comparison in COMPARISON_QUADRANTS:
            if comparison not in pivot.columns:
                continue

            pair = pivot[[REFERENCE_QUADRANT, comparison]].dropna()

            ref_values = pair[REFERENCE_QUADRANT].to_numpy(dtype=float)
            comp_values = pair[comparison].to_numpy(dtype=float)
            diffs = ref_values - comp_values

            mean_ref = float(ref_values.mean())
            mean_comp = float(comp_values.mean())
            mean_diff = float(diffs.mean())

            ci_low, ci_high = bootstrap_ci_mean(diffs)
            p_raw = sign_flip_p_value(diffs)

            num_positive = int((diffs > EPS).sum())
            num_negative = int((diffs < -EPS).sum())
            num_zero = int((np.abs(diffs) <= EPS).sum())

            pairwise_rows.append(
                {
                    "group": group,
                    "reference_quadrant": REFERENCE_QUADRANT,
                    "comparison_quadrant": comparison,
                    "num_paired_queries": len(pair),
                    "mean_reference_rate": mean_ref,
                    "mean_comparison_rate": mean_comp,
                    "mean_difference": mean_diff,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "p_raw_sign_flip": p_raw,
                    "num_ref_higher": num_positive,
                    "num_ref_lower": num_negative,
                    "num_equal": num_zero,
                    "mean_reference_%": round(mean_ref * 100, 4),
                    "mean_comparison_%": round(mean_comp * 100, 4),
                    "mean_difference_%": round(mean_diff * 100, 4),
                    "ci95_low_%": round(ci_low * 100, 4),
                    "ci95_high_%": round(ci_high * 100, 4),
                }
            )

    pairwise_df = pd.DataFrame(pairwise_rows)

    pairwise_df["p_holm"] = holm_bonferroni(
        pairwise_df["p_raw_sign_flip"].tolist()
    )

    pairwise_df["ci_excludes_zero"] = (
        (pairwise_df["ci95_low"] > 0) | (pairwise_df["ci95_high"] < 0)
    )

    pairwise_df["significant_raw_0.05"] = pairwise_df["p_raw_sign_flip"] < 0.05
    pairwise_df["significant_holm_0.05"] = pairwise_df["p_holm"] < 0.05

    pairwise_df["group_order"] = pairwise_df["group"].map(GROUP_ORDER)
    pairwise_df["comparison_order"] = pairwise_df["comparison_quadrant"].map(
        QUADRANT_ORDER
    )

    pairwise_df = pairwise_df.sort_values(
        ["group_order", "comparison_order"]
    ).drop(columns=["group_order", "comparison_order"])

    pairwise_df.to_csv(PAIRWISE_STATS_PATH, index=False)

    md_text = make_markdown_table(pairwise_df)
    PAIRWISE_MD_PATH.write_text(md_text, encoding="utf-8")

    print("\n" + "=" * 80)
    print("Paired query-level comparisons:")
    display_cols = [
        "group",
        "comparison_quadrant",
        "num_paired_queries",
        "mean_reference_%",
        "mean_comparison_%",
        "mean_difference_%",
        "ci95_low_%",
        "ci95_high_%",
        "p_raw_sign_flip",
        "p_holm",
        "ci_excludes_zero",
        "significant_holm_0.05",
    ]
    print(pairwise_df[display_cols].to_string(index=False))

    print("\nSaved files:")
    print(QUERY_LEVEL_RATES_PATH)
    print(SUMMARY_PATH)
    print(PAIRWISE_STATS_PATH)
    print(PAIRWISE_MD_PATH)


if __name__ == "__main__":
    main()