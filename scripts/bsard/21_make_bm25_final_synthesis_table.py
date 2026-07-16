"""Build the final BSARD BM25 synthesis tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TABLE_DIR = Path("outputs/tables/bsard")

OUTPUT_FULL_CSV = TABLE_DIR / "bm25_final_synthesis_table_full.csv"
OUTPUT_FULL_MD = TABLE_DIR / "bm25_final_synthesis_table_full.md"

OUTPUT_MAIN_CSV = TABLE_DIR / "bm25_final_synthesis_table_main.csv"
OUTPUT_MAIN_MD = TABLE_DIR / "bm25_final_synthesis_table_main.md"

# Anciens noms, conservés par compatibilité.
OUTPUT_COMPAT_CSV = TABLE_DIR / "bm25_final_synthesis_table.csv"
OUTPUT_COMPAT_MD = TABLE_DIR / "bm25_final_synthesis_table.md"

METRICS = ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            "Relance les scripts précédents qui produisent ce fichier."
        )


def safe_float(value):
    if value is None:
        return None

    if pd.isna(value):
        return None

    if isinstance(value, str):
        value = value.replace("+", "").strip()
        if value == "":
            return None

    return float(value)


def safe_bool(value):
    """
    Convertit proprement les booléens lus depuis CSV.
    Évite le piège bool("False") == True.
    """
    if value is None or pd.isna(value):
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n"}:
        return False

    return None


def format_float(value, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return ""

    return f"{float(value):.{digits}f}"


def format_delta(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""

    return f"{float(value):+.{digits}f}"


def format_percent(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""

    return f"{float(value):.{digits}f}"


def format_p(value) -> str:
    if value is None or pd.isna(value):
        return ""

    value = float(value)

    if value < 0.001:
        return "<0.001"

    return f"{value:.4f}"


def make_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []

    for _, row in df.iterrows():
        values = [str(row[col]) for col in columns]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *rows])


def load_baseline() -> dict:
    """
    Charge la baseline BM25 originale.

    Source préférée :
      outputs/tables/bsard/rrf_original_reformulation_test_metrics.csv

    Ce fichier contient une ligne bm25_original_test.
    """
    path = TABLE_DIR / "rrf_original_reformulation_test_metrics.csv"
    require_file(path)

    df = pd.read_csv(path)

    row = df[df["method"] == "bm25_original_test"]

    if row.empty:
        raise ValueError(
            f"Could not find method=bm25_original_test in {path}"
        )

    row = row.iloc[0]

    return {metric: float(row[metric]) for metric in METRICS}


def select_long_rows(path: Path, filters: dict) -> pd.DataFrame:
    require_file(path)

    df = pd.read_csv(path)

    selected = df.copy()

    for col, expected in filters.items():
        if col not in selected.columns:
            raise ValueError(f"Column {col!r} not found in {path}")

        selected = selected[selected[col].astype(str) == str(expected)]

    if selected.empty:
        raise ValueError(
            f"No rows found in {path} with filters={filters}"
        )

    return selected


def get_metric_row(df: pd.DataFrame, metric: str) -> pd.Series:
    row = df[df["metric"] == metric]

    if row.empty:
        raise ValueError(f"Missing metric {metric} in selected rows")

    return row.iloc[0]


def read_stats_p_holm(
    path: Path | None,
    filters: dict | None,
    metric: str,
) -> tuple[float | None, bool | None]:
    if path is None or filters is None:
        return None, None

    if not path.exists():
        return None, None

    df = pd.read_csv(path)

    selected = df.copy()

    for col, expected in filters.items():
        if col not in selected.columns:
            return None, None

        selected = selected[selected[col].astype(str) == str(expected)]

    if "metric" not in selected.columns:
        return None, None

    selected = selected[selected["metric"] == metric]

    if selected.empty:
        return None, None

    row = selected.iloc[0]

    p_holm = row["p_holm"] if "p_holm" in row.index else None
    significant = (
        row["significant_holm_0.05"]
        if "significant_holm_0.05" in row.index
        else None
    )

    if pd.isna(p_holm):
        p_holm = None
    else:
        p_holm = float(p_holm)

    significant = safe_bool(significant)

    return p_holm, significant


def add_baseline_row(rows: list[dict], baseline: dict) -> None:
    rows.append(
        {
            "row_order": 10,
            "include_in_main": True,
            "family": "Baseline",
            "method": "BM25 original",
            "setting": "original query only",
            "Recall@10": baseline["Recall@10"],
            "Delta Recall@10": None,
            "Recall@100": baseline["Recall@100"],
            "Delta Recall@100": None,
            "Harm Recall@100 %": None,
            "pHolm Recall@100": None,
            "MRR@10": baseline["MRR@10"],
            "Delta MRR@10": None,
            "pHolm MRR@10": None,
            "nDCG@10": baseline["nDCG@10"],
            "Delta nDCG@10": None,
            "Harm nDCG@10 %": None,
            "pHolm nDCG@10": None,
            "holm significant metrics": "",
            "interpretation": "Canonical lexical baseline.",
        }
    )


def add_from_long_summary(
    rows: list[dict],
    *,
    row_order: int,
    include_in_main: bool,
    family: str,
    method: str,
    setting: str,
    summary_path: Path,
    filters: dict,
    score_col: str,
    baseline: dict,
    stats_path: Path | None = None,
    stats_filters: dict | None = None,
    interpretation: str,
) -> None:
    selected = select_long_rows(summary_path, filters)

    output = {
        "row_order": row_order,
        "include_in_main": include_in_main,
        "family": family,
        "method": method,
        "setting": setting,
        "interpretation": interpretation,
    }

    significant_metrics = []

    for metric in METRICS:
        metric_row = get_metric_row(selected, metric)

        score = safe_float(metric_row[score_col])

        delta = score - baseline[metric]

        harm = None

        if "harm_%" in metric_row.index:
            harm = safe_float(metric_row["harm_%"])

        p_holm, significant = read_stats_p_holm(
            stats_path,
            stats_filters,
            metric,
        )

        output[metric] = score
        output[f"Delta {metric}"] = delta

        if metric in ["Recall@100", "nDCG@10"]:
            output[f"Harm {metric} %"] = harm

        if metric in ["Recall@100", "MRR@10", "nDCG@10"]:
            output[f"pHolm {metric}"] = p_holm

        if significant:
            significant_metrics.append(metric)

    output["holm significant metrics"] = ", ".join(significant_metrics)

    rows.append(
        {
            "row_order": output["row_order"],
            "include_in_main": output["include_in_main"],
            "family": output["family"],
            "method": output["method"],
            "setting": output["setting"],
            "Recall@10": output["Recall@10"],
            "Delta Recall@10": output["Delta Recall@10"],
            "Recall@100": output["Recall@100"],
            "Delta Recall@100": output["Delta Recall@100"],
            "Harm Recall@100 %": output.get("Harm Recall@100 %"),
            "pHolm Recall@100": output.get("pHolm Recall@100"),
            "MRR@10": output["MRR@10"],
            "Delta MRR@10": output["Delta MRR@10"],
            "pHolm MRR@10": output.get("pHolm MRR@10"),
            "nDCG@10": output["nDCG@10"],
            "Delta nDCG@10": output["Delta nDCG@10"],
            "Harm nDCG@10 %": output.get("Harm nDCG@10 %"),
            "pHolm nDCG@10": output.get("pHolm nDCG@10"),
            "holm significant metrics": output["holm significant metrics"],
            "interpretation": output["interpretation"],
        }
    )


def format_for_markdown(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()

    numeric_cols_6 = ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]
    delta_cols = [
        "Delta Recall@10",
        "Delta Recall@100",
        "Delta MRR@10",
        "Delta nDCG@10",
    ]
    harm_cols = ["Harm Recall@100 %", "Harm nDCG@10 %"]
    p_cols = ["pHolm Recall@100", "pHolm MRR@10", "pHolm nDCG@10"]

    for col in numeric_cols_6:
        display_df[col] = display_df[col].apply(format_float)

    for col in delta_cols:
        display_df[col] = display_df[col].apply(format_delta)

    for col in harm_cols:
        display_df[col] = display_df[col].apply(format_percent)

    for col in p_cols:
        display_df[col] = display_df[col].apply(format_p)

    return display_df


def save_table(df: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    """
    Sauvegarde une version numérique CSV et une version Markdown lisible.
    """
    df_to_save = df.drop(columns=["row_order", "include_in_main"], errors="ignore")
    df_to_save.to_csv(csv_path, index=False)

    display_df = format_for_markdown(df_to_save)

    markdown_text = make_markdown_table(display_df)
    md_path.write_text(markdown_text, encoding="utf-8")


def main() -> None:
    baseline = load_baseline()

    rows = []

    add_baseline_row(rows, baseline)

    # --------------------------------------------------------
    # Replacement direct : reformulation seule
    # --------------------------------------------------------
    replacement_summary = TABLE_DIR / "bm25_reformulations_summary_for_paper.csv"

    add_from_long_summary(
        rows,
        row_order=20,
        include_in_main=True,
        family="Replacement",
        method="DeepSeek keyword alone",
        setting="keyword_expansion replaces original",
        summary_path=replacement_summary,
        filters={"generator": "deepseek", "query_type": "keyword_expansion"},
        score_col="candidate",
        baseline=baseline,
        stats_path=None,
        stats_filters=None,
        interpretation="Improves broad recall but remains top-rank unstable.",
    )

    add_from_long_summary(
        rows,
        row_order=30,
        include_in_main=True,
        family="Replacement",
        method="GPT keyword alone",
        setting="keyword_expansion replaces original",
        summary_path=replacement_summary,
        filters={"generator": "gpt", "query_type": "keyword_expansion"},
        score_col="candidate",
        baseline=baseline,
        stats_path=None,
        stats_filters=None,
        interpretation="Smaller recall gain, weaker top-rank behavior.",
    )

    # --------------------------------------------------------
    # RRF original + une reformulation
    # --------------------------------------------------------
    one_view_summary = TABLE_DIR / "rrf_original_reformulation_summary_for_paper.csv"
    one_view_stats = TABLE_DIR / "rrf_original_reformulation_stats.csv"

    add_from_long_summary(
        rows,
        row_order=40,
        include_in_main=True,
        family="RRF one-view",
        method="Original + DeepSeek keyword",
        setting="original + keyword_expansion",
        summary_path=one_view_summary,
        filters={"generator": "deepseek", "query_type": "keyword_expansion"},
        score_col="fusion_score",
        baseline=baseline,
        stats_path=one_view_stats,
        stats_filters={"generator": "deepseek", "query_type": "keyword_expansion"},
        interpretation="Safest broad-recall strategy.",
    )

    add_from_long_summary(
        rows,
        row_order=50,
        include_in_main=True,
        family="RRF one-view",
        method="Original + GPT keyword",
        setting="original + keyword_expansion",
        summary_path=one_view_summary,
        filters={"generator": "gpt", "query_type": "keyword_expansion"},
        score_col="fusion_score",
        baseline=baseline,
        stats_path=one_view_stats,
        stats_filters={"generator": "gpt", "query_type": "keyword_expansion"},
        interpretation="Positive broad-recall gain, weaker than DeepSeek keyword.",
    )

    # --------------------------------------------------------
    # RRF original + toutes les reformulations
    # --------------------------------------------------------
    all_rrf_summary = TABLE_DIR / "rrf_all_reformulations_summary_for_paper.csv"
    all_rrf_stats = TABLE_DIR / "rrf_all_reformulations_stats.csv"

    add_from_long_summary(
        rows,
        row_order=60,
        include_in_main=False,
        family="RRF multi-view",
        method="Original + all DeepSeek",
        setting="original + 3 DeepSeek reformulations",
        summary_path=all_rrf_summary,
        filters={"generator": "deepseek", "fusion_type": "original_plus_all_deepseek"},
        score_col="fusion_score",
        baseline=baseline,
        stats_path=all_rrf_stats,
        stats_filters={"generator": "deepseek", "fusion_type": "original_plus_all_deepseek"},
        interpretation="Robust gains for Recall@100 and MRR@10.",
    )

    add_from_long_summary(
        rows,
        row_order=70,
        include_in_main=False,
        family="RRF multi-view",
        method="Original + all GPT",
        setting="original + 3 GPT reformulations",
        summary_path=all_rrf_summary,
        filters={"generator": "gpt", "fusion_type": "original_plus_all_gpt"},
        score_col="fusion_score",
        baseline=baseline,
        stats_path=all_rrf_stats,
        stats_filters={"generator": "gpt", "fusion_type": "original_plus_all_gpt"},
        interpretation="Positive trend but weaker robustness.",
    )

    add_from_long_summary(
        rows,
        row_order=80,
        include_in_main=True,
        family="RRF multi-view",
        method="Original + all DeepSeek+GPT",
        setting="original + 6 reformulations",
        summary_path=all_rrf_summary,
        filters={"generator": "deepseek+gpt", "fusion_type": "original_plus_all_generators"},
        score_col="fusion_score",
        baseline=baseline,
        stats_path=all_rrf_stats,
        stats_filters={"generator": "deepseek+gpt", "fusion_type": "original_plus_all_generators"},
        interpretation="Best RRF balance between recall and top-rank.",
    )

    # --------------------------------------------------------
    # RCS selected configurations
    # --------------------------------------------------------
    rcs_summary = TABLE_DIR / "rcs_selected_grid_summary_for_paper.csv"
    rcs_stats = TABLE_DIR / "rcs_selected_grid_stats.csv"

    add_from_long_summary(
        rows,
        row_order=90,
        include_in_main=True,
        family="RCS selected configurations",
        method="DeepSeek top-rank RCS",
        setting="alpha=1, beta=0.5, gamma=2, min_votes=2",
        summary_path=rcs_summary,
        filters={"label": "rcs_deepseek_best_toprank"},
        score_col="candidate",
        baseline=baseline,
        stats_path=rcs_stats,
        stats_filters={"label": "rcs_deepseek_best_toprank"},
        interpretation="Best top-rank configuration; consensus bonus helps.",
    )

    add_from_long_summary(
        rows,
        row_order=100,
        include_in_main=False,
        family="RCS selected configurations",
        method="All generators RRF-like",
        setting="alpha=1, beta=1, gamma=0, min_votes=2",
        summary_path=rcs_summary,
        filters={"label": "rcs_all_rrf_like_best_recall"},
        score_col="candidate",
        baseline=baseline,
        stats_path=rcs_stats,
        stats_filters={"label": "rcs_all_rrf_like_best_recall"},
        interpretation="Best all-generator Recall@100; no consensus bonus.",
    )

    add_from_long_summary(
        rows,
        row_order=110,
        include_in_main=True,
        family="RCS selected configurations",
        method="All generators strong consensus",
        setting="alpha=1, beta=1, gamma=2, min_votes=2",
        summary_path=rcs_summary,
        filters={"label": "rcs_all_consensus_strong"},
        score_col="candidate",
        baseline=baseline,
        stats_path=rcs_stats,
        stats_filters={"label": "rcs_all_consensus_strong"},
        interpretation="Consensus bonus improves nDCG@10 robustness.",
    )

    final_df = pd.DataFrame(rows).sort_values("row_order")

    main_df = final_df[final_df["include_in_main"]].copy()

    # Sauvegarde des deux versions.
    save_table(final_df, OUTPUT_FULL_CSV, OUTPUT_FULL_MD)
    save_table(main_df, OUTPUT_MAIN_CSV, OUTPUT_MAIN_MD)

    # Compatibilité avec les anciens noms : version complète.
    save_table(final_df, OUTPUT_COMPAT_CSV, OUTPUT_COMPAT_MD)

    print("\nBM25 final synthesis table — MAIN:")
    main_display = format_for_markdown(
        main_df.drop(columns=["row_order", "include_in_main"], errors="ignore")
    )
    print(main_display.to_string(index=False))

    print("\nBM25 final synthesis table — FULL:")
    full_display = format_for_markdown(
        final_df.drop(columns=["row_order", "include_in_main"], errors="ignore")
    )
    print(full_display.to_string(index=False))

    print("\nSaved files:")
    print(OUTPUT_MAIN_CSV)
    print(OUTPUT_MAIN_MD)
    print(OUTPUT_FULL_CSV)
    print(OUTPUT_FULL_MD)
    print(OUTPUT_COMPAT_CSV)
    print(OUTPUT_COMPAT_MD)


if __name__ == "__main__":
    main()
