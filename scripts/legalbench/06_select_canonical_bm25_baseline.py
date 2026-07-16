"""Select and export the canonical LegalBench BM25 baseline."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


RUNS_DIR = Path("runs/legalbench")
OUT_DIR = Path("outputs/tables/legalbench")

CANONICAL_VARIANT = "filepath_plus_chunk"
SOURCE_METHOD = f"bm25_original_{CANONICAL_VARIANT}"
CANONICAL_METHOD = "bm25_original_mini_canonical"

SOURCE_RUN = RUNS_DIR / f"{SOURCE_METHOD}_mini.tsv"
CANONICAL_RUN = RUNS_DIR / f"{CANONICAL_METHOD}.tsv"

VARIANT_METRICS_PATH = OUT_DIR / "bm25_original_text_variants_mini_metrics.csv"
VARIANT_BY_TASK_PATH = OUT_DIR / "bm25_original_text_variants_mini_by_task.csv"
VARIANT_PER_QUERY_PATH = OUT_DIR / "bm25_original_text_variants_mini_per_query.csv"

CANONICAL_METRICS_PATH = OUT_DIR / "bm25_original_mini_canonical_metrics.csv"
CANONICAL_BY_TASK_PATH = OUT_DIR / "bm25_original_mini_canonical_by_task.csv"
CANONICAL_PER_QUERY_PATH = OUT_DIR / "bm25_original_mini_canonical_per_query.csv"
CANONICAL_SUMMARY_MD_PATH = OUT_DIR / "bm25_original_mini_canonical_summary.md"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            "Relance d'abord le script 05 :\n"
            "  python scripts/legalbench/05_bm25_legalbench_original_text_variants.py"
        )


def make_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []

    for _, row in df.iterrows():
        values = [str(row[col]) for col in columns]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *rows])


def select_rows(
    *,
    path: Path,
    variant: str,
    output_method: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "text_variant" not in df.columns:
        raise ValueError(f"Column text_variant not found in {path}")

    selected = df[df["text_variant"] == variant].copy()

    if selected.empty:
        raise ValueError(
            f"No rows found for text_variant={variant!r} in {path}"
        )

    if "method" in selected.columns:
        selected["source_method"] = selected["method"]
        selected["method"] = output_method

    selected["canonical_text_variant"] = variant

    return selected


def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Selecting canonical BM25 baseline for LegalBench-RAG mini")
    print("=" * 80)

    print(f"Canonical variant: {CANONICAL_VARIANT}")
    print(f"Source method: {SOURCE_METHOD}")
    print(f"Canonical method: {CANONICAL_METHOD}")

    require_file(SOURCE_RUN)
    require_file(VARIANT_METRICS_PATH)
    require_file(VARIANT_BY_TASK_PATH)
    require_file(VARIANT_PER_QUERY_PATH)

    print("\nCopying canonical run...")
    shutil.copyfile(SOURCE_RUN, CANONICAL_RUN)
    print(f"Copied:")
    print(f"  from: {SOURCE_RUN}")
    print(f"  to:   {CANONICAL_RUN}")

    print("\nExtracting canonical metrics...")
    metrics_df = select_rows(
        path=VARIANT_METRICS_PATH,
        variant=CANONICAL_VARIANT,
        output_method=CANONICAL_METHOD,
    )

    by_task_df = select_rows(
        path=VARIANT_BY_TASK_PATH,
        variant=CANONICAL_VARIANT,
        output_method=CANONICAL_METHOD,
    )

    per_query_df = select_rows(
        path=VARIANT_PER_QUERY_PATH,
        variant=CANONICAL_VARIANT,
        output_method=CANONICAL_METHOD,
    )

    metrics_df.to_csv(CANONICAL_METRICS_PATH, index=False)
    by_task_df.to_csv(CANONICAL_BY_TASK_PATH, index=False)
    per_query_df.to_csv(CANONICAL_PER_QUERY_PATH, index=False)

    print(f"Saved canonical global metrics to: {CANONICAL_METRICS_PATH}")
    print(f"Saved canonical by-task metrics to: {CANONICAL_BY_TASK_PATH}")
    print(f"Saved canonical per-query metrics to: {CANONICAL_PER_QUERY_PATH}")

    summary_lines = [
        "# LegalBench-RAG mini — canonical BM25 baseline",
        "",
        f"Canonical method: `{CANONICAL_METHOD}`",
        "",
        f"Selected source method: `{SOURCE_METHOD}`",
        "",
        f"Selected text variant: `{CANONICAL_VARIANT}`",
        "",
        "## Global metrics",
        "",
        make_markdown_table(metrics_df),
        "",
        "## Metrics by task",
        "",
        make_markdown_table(by_task_df),
        "",
        "## Methodological note",
        "",
        (
            "The canonical BM25 baseline indexes each chunk together with "
            "its relative file path. This is used because LegalBench-RAG "
            "queries often identify the target document or policy, while "
            "the answer-bearing chunk may not repeat the document name."
        ),
        "",
    ]

    CANONICAL_SUMMARY_MD_PATH.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print(f"Saved markdown summary to: {CANONICAL_SUMMARY_MD_PATH}")

    print("\n" + "=" * 80)
    print("Canonical BM25 baseline — global metrics")
    print("=" * 80)
    print(metrics_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Canonical BM25 baseline — metrics by task")
    print("=" * 80)
    print(by_task_df.to_string(index=False))

    print("\nSaved files:")
    print(CANONICAL_RUN)
    print(CANONICAL_METRICS_PATH)
    print(CANONICAL_BY_TASK_PATH)
    print(CANONICAL_PER_QUERY_PATH)
    print(CANONICAL_SUMMARY_MD_PATH)

    print("\nNext:")
    print("  scripts/legalbench/07_prepare_queries_for_reformulation.py")


if __name__ == "__main__":
    main()