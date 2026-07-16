"""Summarize BSARD BM25 replacement results across reformulation methods."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TABLE_DIR = Path("outputs/tables/bsard")

INPUT_PATH = TABLE_DIR / "bm25_reformulations_harm_rate.csv"
OUTPUT_CSV = TABLE_DIR / "bm25_reformulations_summary_for_paper.csv"
OUTPUT_MD = TABLE_DIR / "bm25_reformulations_summary_for_paper.md"


METRIC_ORDER = {
    "Recall@10": 1,
    "Recall@100": 2,
    "MRR@10": 3,
    "nDCG@10": 4,
}

QUERY_TYPE_ORDER = {
    "legal_rewrite": 1,
    "keyword_expansion": 2,
    "hyde_style": 3,
}

GENERATOR_ORDER = {
    "deepseek": 1,
    "gpt": 2,
}


def format_signed(value: float, digits: int = 4) -> str:
    """
    Formate un nombre avec signe.
    Exemple :
        +0.0510
        -0.0177
    """
    return f"{value:+.{digits}f}"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    print(f"Loading harm-rate table: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "generator",
        "query_type",
        "metric",
        "mean_baseline",
        "mean_candidate",
        "mean_gain",
        "harm_rate",
        "improve_rate",
        "neutral_rate",
        "num_improved",
        "num_harmed",
        "num_neutral",
        "num_queries",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input file: {missing}")

    summary = df.copy()

    # Colonnes numériques lisibles
    summary["baseline"] = summary["mean_baseline"].round(6)
    summary["candidate"] = summary["mean_candidate"].round(6)
    summary["delta"] = summary["mean_gain"].round(6)

    summary["delta_display"] = summary["mean_gain"].apply(format_signed)

    summary["harm_%"] = (summary["harm_rate"] * 100).round(2)
    summary["improve_%"] = (summary["improve_rate"] * 100).round(2)
    summary["neutral_%"] = (summary["neutral_rate"] * 100).round(2)

    summary["improved/harmed/neutral"] = (
        summary["num_improved"].astype(str)
        + "/"
        + summary["num_harmed"].astype(str)
        + "/"
        + summary["num_neutral"].astype(str)
    )

    summary["generator_order"] = summary["generator"].map(GENERATOR_ORDER)
    summary["query_type_order"] = summary["query_type"].map(QUERY_TYPE_ORDER)
    summary["metric_order"] = summary["metric"].map(METRIC_ORDER)

    summary = summary.sort_values(
        ["generator_order", "query_type_order", "metric_order"]
    )

    final_columns = [
        "generator",
        "query_type",
        "metric",
        "baseline",
        "candidate",
        "delta_display",
        "harm_%",
        "improve_%",
        "neutral_%",
        "improved/harmed/neutral",
    ]

    final = summary[final_columns].rename(
        columns={
            "delta_display": "delta",
        }
    )

    final.to_csv(OUTPUT_CSV, index=False)

    # Sauvegarde Markdown pour copier facilement dans notes/papier
    markdown_text = final.to_markdown(index=False)
    OUTPUT_MD.write_text(markdown_text, encoding="utf-8")

    print("\nSummary table:")
    print(final.to_string(index=False))

    print(f"\nSaved CSV to: {OUTPUT_CSV}")
    print(f"Saved Markdown to: {OUTPUT_MD}")

    print("\nKey observations:")
    recall100 = final[final["metric"] == "Recall@100"].copy()
    ndcg10 = final[final["metric"] == "nDCG@10"].copy()

    print("\nRecall@100 rows:")
    print(recall100.to_string(index=False))

    print("\nnDCG@10 rows:")
    print(ndcg10.to_string(index=False))


if __name__ == "__main__":
    main()