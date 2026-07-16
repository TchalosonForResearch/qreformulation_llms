"""Run paired statistical tests for one-view BSARD RRF configurations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TABLE_DIR = Path("outputs/tables/bsard")

INPUT_PATH = TABLE_DIR / "rrf_original_reformulation_per_query_all.csv"
OUTPUT_PATH = TABLE_DIR / "rrf_original_reformulation_stats.csv"

METRICS = ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]

N_BOOTSTRAP = 10000
N_RANDOMIZATION = 10000
RANDOM_SEED = 42
EPS = 1e-12


def bootstrap_ci_mean(
    values: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """
    Calcule un intervalle de confiance bootstrap percentile à 95 %
    pour la moyenne des gains.
    """
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
    """
    Sign-flip randomization test bilatéral.

    Hypothèse nulle :
        le signe des gains est arbitraire.

    On garde les magnitudes des gains, mais on inverse aléatoirement
    leurs signes. Si le gain moyen observé est rare sous ces inversions,
    la p-value est petite.
    """
    rng = np.random.default_rng(seed)

    observed = abs(float(gains.mean()))
    n = len(gains)

    count = 0

    for _ in range(n_randomization):
        signs = rng.choice([-1.0, 1.0], size=n, replace=True)
        randomized_mean = abs(float((gains * signs).mean()))

        if randomized_mean >= observed - EPS:
            count += 1

    # +1 smoothing pour éviter p=0
    return float((count + 1) / (n_randomization + 1))


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """
    Correction Holm-Bonferroni.

    Retourne des p-values ajustées monotones.
    """
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


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    print(f"Loading per-query gains: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, dtype={"query_id": str})

    required = {"generator", "query_type", "query_id"}

    for metric in METRICS:
        required.add(f"{metric}_baseline")
        required.add(f"{metric}_candidate")
        required.add(f"{metric}_gain")

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    rows = []

    grouped = df.groupby(["generator", "query_type"], sort=False)

    for (generator, query_type), group in grouped:
        for metric in METRICS:
            gains = group[f"{metric}_gain"].to_numpy(dtype=float)

            mean_baseline = float(group[f"{metric}_baseline"].mean())
            mean_candidate = float(group[f"{metric}_candidate"].mean())
            mean_gain = float(gains.mean())

            ci_low, ci_high = bootstrap_ci_mean(
                gains,
                n_bootstrap=N_BOOTSTRAP,
                seed=RANDOM_SEED,
            )

            p_raw = sign_flip_p_value(
                gains,
                n_randomization=N_RANDOMIZATION,
                seed=RANDOM_SEED,
            )

            num_positive = int((gains > EPS).sum())
            num_negative = int((gains < -EPS).sum())
            num_zero = int((np.abs(gains) <= EPS).sum())

            rows.append(
                {
                    "generator": generator,
                    "query_type": query_type,
                    "metric": metric,
                    "num_queries": len(gains),
                    "mean_baseline": mean_baseline,
                    "mean_candidate": mean_candidate,
                    "mean_gain": mean_gain,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "p_raw_sign_flip": p_raw,
                    "num_positive": num_positive,
                    "num_negative": num_negative,
                    "num_zero": num_zero,
                }
            )

    stats_df = pd.DataFrame(rows)

    stats_df["p_holm"] = holm_bonferroni(
        stats_df["p_raw_sign_flip"].tolist()
    )

    stats_df["significant_raw_0.05"] = stats_df["p_raw_sign_flip"] < 0.05
    stats_df["significant_holm_0.05"] = stats_df["p_holm"] < 0.05
    stats_df["ci_excludes_zero"] = (
        (stats_df["ci95_low"] > 0) | (stats_df["ci95_high"] < 0)
    )

    query_type_order = {
        "legal_rewrite": 1,
        "keyword_expansion": 2,
        "hyde_style": 3,
    }

    metric_order = {
        "Recall@10": 1,
        "Recall@100": 2,
        "MRR@10": 3,
        "nDCG@10": 4,
    }

    generator_order = {
        "deepseek": 1,
        "gpt": 2,
    }

    stats_df["generator_order"] = stats_df["generator"].map(generator_order)
    stats_df["query_type_order"] = stats_df["query_type"].map(query_type_order)
    stats_df["metric_order"] = stats_df["metric"].map(metric_order)

    stats_df = stats_df.sort_values(
        ["generator_order", "query_type_order", "metric_order"]
    ).drop(
        columns=["generator_order", "query_type_order", "metric_order"]
    )

    stats_df.to_csv(OUTPUT_PATH, index=False)

    print("\nStatistical summary:")
    display_cols = [
        "generator",
        "query_type",
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

    print(f"\nSaved stats to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()