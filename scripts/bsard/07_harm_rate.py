"""Compute query-level gains and harm rates for BSARD replacement runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TABLE_DIR = Path("outputs/tables/bsard")

BASELINE_PATH = TABLE_DIR / "bm25_original_test_per_query.csv"

CANDIDATES = [
    ("deepseek", "legal_rewrite", TABLE_DIR / "bm25_deepseek_legal_rewrite_test_per_query.csv"),
    ("deepseek", "keyword_expansion", TABLE_DIR / "bm25_deepseek_keyword_expansion_test_per_query.csv"),
    ("deepseek", "hyde_style", TABLE_DIR / "bm25_deepseek_hyde_style_test_per_query.csv"),
    ("gpt", "legal_rewrite", TABLE_DIR / "bm25_gpt_legal_rewrite_test_per_query.csv"),
    ("gpt", "keyword_expansion", TABLE_DIR / "bm25_gpt_keyword_expansion_test_per_query.csv"),
    ("gpt", "hyde_style", TABLE_DIR / "bm25_gpt_hyde_style_test_per_query.csv"),
]

METRICS = ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]


def load_per_query(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing per-query file: {path}")

    df = pd.read_csv(path, dtype={"query_id": str})
    required = {"query_id", *METRICS}

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    return df


def classify_gain(gain: float) -> str:
    """
    Classe le résultat pour une requête :
    - improved : gain > 0
    - harmed   : gain < 0
    - neutral  : gain == 0
    """
    if gain > 0:
        return "improved"
    if gain < 0:
        return "harmed"
    return "neutral"


def main() -> None:
    print(f"Loading baseline: {BASELINE_PATH}")
    baseline = load_per_query(BASELINE_PATH)

    baseline = baseline[["query_id", *METRICS]].copy()
    baseline = baseline.rename(columns={m: f"{m}_baseline" for m in METRICS})

    summary_rows = []
    gain_rows = []

    for generator, query_type, candidate_path in CANDIDATES:
        print("\n" + "=" * 80)
        print(f"Candidate: generator={generator}, query_type={query_type}")
        print(f"File: {candidate_path}")

        candidate = load_per_query(candidate_path)
        candidate = candidate[["query_id", *METRICS]].copy()
        candidate = candidate.rename(columns={m: f"{m}_candidate" for m in METRICS})

        merged = baseline.merge(candidate, on="query_id", how="inner")

        if len(merged) != len(baseline):
            raise ValueError(
                f"Query mismatch for {generator}/{query_type}: "
                f"{len(merged)} merged vs {len(baseline)} baseline"
            )

        for metric in METRICS:
            base_col = f"{metric}_baseline"
            cand_col = f"{metric}_candidate"
            gain_col = f"{metric}_gain"
            status_col = f"{metric}_status"

            merged[gain_col] = merged[cand_col] - merged[base_col]
            merged[status_col] = merged[gain_col].apply(classify_gain)

            num_queries = len(merged)
            num_harmed = int((merged[gain_col] < 0).sum())
            num_improved = int((merged[gain_col] > 0).sum())
            num_neutral = int((merged[gain_col] == 0).sum())

            summary_rows.append(
                {
                    "generator": generator,
                    "query_type": query_type,
                    "metric": metric,
                    "mean_baseline": float(merged[base_col].mean()),
                    "mean_candidate": float(merged[cand_col].mean()),
                    "mean_gain": float(merged[gain_col].mean()),
                    "num_queries": num_queries,
                    "num_improved": num_improved,
                    "num_harmed": num_harmed,
                    "num_neutral": num_neutral,
                    "improve_rate": num_improved / num_queries,
                    "harm_rate": num_harmed / num_queries,
                    "neutral_rate": num_neutral / num_queries,
                }
            )

        merged.insert(1, "generator", generator)
        merged.insert(2, "query_type", query_type)
        gain_rows.append(merged)

    summary_df = pd.DataFrame(summary_rows)

    gains_df = pd.concat(gain_rows, ignore_index=True)

    summary_path = TABLE_DIR / "bm25_reformulations_harm_rate.csv"
    gains_path = TABLE_DIR / "bm25_reformulations_per_query_gains.csv"

    summary_df.to_csv(summary_path, index=False)
    gains_df.to_csv(gains_path, index=False)

    print("\n" + "=" * 80)
    print("Harm-rate summary:")
    print(
        summary_df.sort_values(["metric", "generator", "query_type"])
        .to_string(index=False)
    )

    print(f"\nSaved harm-rate summary to: {summary_path}")
    print(f"Saved per-query gains to: {gains_path}")


if __name__ == "__main__":
    main()