"""Build the final LegalBench result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("outputs/tables/legalbench")

BASELINE_METRICS_PATH = OUT_DIR / "bm25_original_mini_canonical_metrics.csv"
BASELINE_PER_QUERY_PATH = OUT_DIR / "bm25_original_mini_canonical_per_query.csv"

REPLACEMENT_PER_QUERY_PATH = OUT_DIR / "bm25_reformulations_mini_per_query_all.csv"

RRF_ONE_VIEW_SUMMARY_PATH = OUT_DIR / "rrf_original_reformulation_mini_summary_for_paper.csv"
MULTIVIEW_SUMMARY_PATH = OUT_DIR / "multiview_rrf_rcs_mini_summary_for_paper.csv"

OUT_REPLACEMENT_SUMMARY = OUT_DIR / "legalbench_replacement_summary_for_synthesis.csv"

OUT_MAIN_CSV = OUT_DIR / "legalbench_final_synthesis_table_main.csv"
OUT_FULL_CSV = OUT_DIR / "legalbench_final_synthesis_table_full.csv"
OUT_MAIN_MD = OUT_DIR / "legalbench_final_synthesis_table_main.md"
OUT_FULL_MD = OUT_DIR / "legalbench_final_synthesis_table_full.md"


METRICS = ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]


def parse_float(value) -> float:
    if value is None:
        return np.nan

    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip()

    if text == "" or text.lower() in {"nan", "none"}:
        return np.nan

    text = text.replace("%", "").replace(",", "")
    text = text.replace("+", "")

    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    return text in {"true", "1", "yes", "y"}


def fmt_score(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.6f}"


def fmt_delta(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):+.4f}"


def fmt_pct(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def fmt_p(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


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


def build_replacement_summary(
    *,
    eps: float,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    """
    Construit les stats harm/pHolm pour les reformulations seules,
    car elles n'avaient pas encore leur table finale dédiée.
    """
    if not BASELINE_PER_QUERY_PATH.exists():
        raise FileNotFoundError(f"Missing file: {BASELINE_PER_QUERY_PATH}")

    if not REPLACEMENT_PER_QUERY_PATH.exists():
        raise FileNotFoundError(f"Missing file: {REPLACEMENT_PER_QUERY_PATH}")

    baseline = pd.read_csv(BASELINE_PER_QUERY_PATH, dtype={"query_id": str})
    replacement = pd.read_csv(REPLACEMENT_PER_QUERY_PATH, dtype={"query_id": str})

    baseline = baseline[["query_id", "task", *METRICS]].copy()
    baseline = baseline.rename(
        columns={metric: f"baseline_{metric}" for metric in METRICS}
    )

    merged = replacement.merge(
        baseline,
        on=["query_id", "task"],
        how="left",
        validate="many_to_one",
    )

    for metric in METRICS:
        merged[f"gain_{metric}"] = merged[metric] - merged[f"baseline_{metric}"]
        merged[f"status_{metric}"] = merged[f"gain_{metric}"].apply(
            lambda x: classify_gain(float(x), eps)
        )

    rows = []
    p_rows = []
    counter = 0

    group_cols = ["method", "generator", "query_type"]

    for group_key, group in merged.groupby(group_cols):
        group_dict = dict(zip(group_cols, group_key))

        for metric in METRICS:
            gain_col = f"gain_{metric}"
            status_col = f"status_{metric}"
            baseline_col = f"baseline_{metric}"

            values = group[gain_col].to_numpy(dtype=float)

            num_queries = int(len(group))
            num_improved = int((group[status_col] == "improved").sum())
            num_harmed = int((group[status_col] == "harmed").sum())
            num_neutral = int((group[status_col] == "neutral").sum())

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

            p_rows.append(p_raw)

            rows.append(
                {
                    **group_dict,
                    "source": "replacement",
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
                    "improved/harmed/neutral": f"{num_improved}/{num_harmed}/{num_neutral}",
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "p_raw_sign_flip": p_raw,
                    "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
                }
            )

            counter += 1

    out = pd.DataFrame(rows)
    out["p_holm"] = holm_correction(p_rows)
    out["significant_holm_0.05"] = out["p_holm"] < 0.05

    out.to_csv(OUT_REPLACEMENT_SUMMARY, index=False)

    return out


def standardize_metric_summary(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    candidate_col = "candidate"

    if candidate_col not in df.columns:
        if "fusion_score" in df.columns:
            candidate_col = "fusion_score"
        elif "rcs_score" in df.columns:
            candidate_col = "rcs_score"
        else:
            raise ValueError(f"No candidate/fusion_score column found in {path}")

    required = {"method", "metric", "baseline", candidate_col, "delta"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    rows = []

    for _, row in df.iterrows():
        method = str(row["method"])
        metric = str(row["metric"])

        rows.append(
            {
                "method": method,
                "source": source,
                "metric": metric,
                "baseline": parse_float(row.get("baseline")),
                "candidate": parse_float(row.get(candidate_col)),
                "delta": parse_float(row.get("delta")),
                "harm_%": parse_float(row.get("harm_%")),
                "improve_%": parse_float(row.get("improve_%")),
                "neutral_%": parse_float(row.get("neutral_%")),
                "p_holm": parse_float(row.get("p_holm")),
                "ci95_low": parse_float(row.get("ci95_low")),
                "ci95_high": parse_float(row.get("ci95_high")),
                "significant_holm_0.05": parse_bool(row.get("significant_holm_0.05")),
                "family_from_file": row.get("family", ""),
                "fusion_type_from_file": row.get("fusion_type", ""),
                "views_from_file": row.get("views", ""),
            }
        )

    return pd.DataFrame(rows)


def load_baseline_row() -> dict:
    if not BASELINE_METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {BASELINE_METRICS_PATH}")

    df = pd.read_csv(BASELINE_METRICS_PATH)

    if df.empty:
        raise ValueError(f"Empty baseline metrics file: {BASELINE_METRICS_PATH}")

    row = df.iloc[0].to_dict()

    return {
        "Recall@10": parse_float(row.get("Recall@10")),
        "Recall@100": parse_float(row.get("Recall@100")),
        "MRR@10": parse_float(row.get("MRR@10")),
        "nDCG@10": parse_float(row.get("nDCG@10")),
    }


def make_lookup(metric_rows: pd.DataFrame) -> dict[tuple[str, str], dict]:
    lookup = {}

    for _, row in metric_rows.iterrows():
        lookup[(str(row["method"]), str(row["metric"]))] = row.to_dict()

    return lookup


def metric_value(
    *,
    lookup: dict[tuple[str, str], dict],
    method: str,
    metric: str,
    field: str,
) -> float:
    row = lookup.get((method, metric))

    if row is None:
        return np.nan

    return parse_float(row.get(field))


def metric_bool(
    *,
    lookup: dict[tuple[str, str], dict],
    method: str,
    metric: str,
    field: str,
) -> bool:
    row = lookup.get((method, metric))

    if row is None:
        return False

    return parse_bool(row.get(field))


def significant_metrics_for_method(
    *,
    lookup: dict[tuple[str, str], dict],
    method: str,
) -> tuple[str, str]:
    positive = []
    negative = []

    for metric in METRICS:
        row = lookup.get((method, metric))

        if row is None:
            continue

        significant = parse_bool(row.get("significant_holm_0.05"))
        delta = parse_float(row.get("delta"))

        if significant and delta > 0:
            positive.append(metric)

        if significant and delta < 0:
            negative.append(metric)

    return ", ".join(positive), ", ".join(negative)


def build_table(
    *,
    configs: list[dict],
    lookup: dict[tuple[str, str], dict],
    baseline_scores: dict,
) -> pd.DataFrame:
    rows = []

    for cfg in configs:
        method_key = cfg.get("method_key", "")

        if cfg["family"] == "Baseline":
            row = {
                "family": cfg["family"],
                "method": cfg["method"],
                "setting": cfg["setting"],
                "Recall@10": fmt_score(baseline_scores["Recall@10"]),
                "Delta Recall@10": "",
                "pHolm Recall@10": "",
                "Recall@100": fmt_score(baseline_scores["Recall@100"]),
                "Delta Recall@100": "",
                "Harm Recall@100 %": "",
                "pHolm Recall@100": "",
                "MRR@10": fmt_score(baseline_scores["MRR@10"]),
                "Delta MRR@10": "",
                "pHolm MRR@10": "",
                "nDCG@10": fmt_score(baseline_scores["nDCG@10"]),
                "Delta nDCG@10": "",
                "Harm nDCG@10 %": "",
                "pHolm nDCG@10": "",
                "Holm significant positive metrics": "",
                "Holm significant negative metrics": "",
                "interpretation": cfg["interpretation"],
            }

            rows.append(row)
            continue

        positive_sig, negative_sig = significant_metrics_for_method(
            lookup=lookup,
            method=method_key,
        )

        row = {
            "family": cfg["family"],
            "method": cfg["method"],
            "setting": cfg["setting"],
            "Recall@10": fmt_score(metric_value(lookup=lookup, method=method_key, metric="Recall@10", field="candidate")),
            "Delta Recall@10": fmt_delta(metric_value(lookup=lookup, method=method_key, metric="Recall@10", field="delta")),
            "pHolm Recall@10": fmt_p(metric_value(lookup=lookup, method=method_key, metric="Recall@10", field="p_holm")),
            "Recall@100": fmt_score(metric_value(lookup=lookup, method=method_key, metric="Recall@100", field="candidate")),
            "Delta Recall@100": fmt_delta(metric_value(lookup=lookup, method=method_key, metric="Recall@100", field="delta")),
            "Harm Recall@100 %": fmt_pct(metric_value(lookup=lookup, method=method_key, metric="Recall@100", field="harm_%")),
            "pHolm Recall@100": fmt_p(metric_value(lookup=lookup, method=method_key, metric="Recall@100", field="p_holm")),
            "MRR@10": fmt_score(metric_value(lookup=lookup, method=method_key, metric="MRR@10", field="candidate")),
            "Delta MRR@10": fmt_delta(metric_value(lookup=lookup, method=method_key, metric="MRR@10", field="delta")),
            "pHolm MRR@10": fmt_p(metric_value(lookup=lookup, method=method_key, metric="MRR@10", field="p_holm")),
            "nDCG@10": fmt_score(metric_value(lookup=lookup, method=method_key, metric="nDCG@10", field="candidate")),
            "Delta nDCG@10": fmt_delta(metric_value(lookup=lookup, method=method_key, metric="nDCG@10", field="delta")),
            "Harm nDCG@10 %": fmt_pct(metric_value(lookup=lookup, method=method_key, metric="nDCG@10", field="harm_%")),
            "pHolm nDCG@10": fmt_p(metric_value(lookup=lookup, method=method_key, metric="nDCG@10", field="p_holm")),
            "Holm significant positive metrics": positive_sig,
            "Holm significant negative metrics": negative_sig,
            "interpretation": cfg["interpretation"],
        }

        rows.append(row)

    return pd.DataFrame(rows)


def write_markdown(path: Path, title: str, df: pd.DataFrame) -> None:
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(df.to_markdown(index=False))
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build LegalBench final synthesis tables."
    )

    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("=" * 80)
    print("Building LegalBench final synthesis tables")
    print("=" * 80)

    baseline_scores = load_baseline_row()

    print("Computing replacement summary for synthesis...")
    replacement_summary = build_replacement_summary(
        eps=args.eps,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )

    print("Loading one-view RRF summary...")
    rrf_one_summary = standardize_metric_summary(
        RRF_ONE_VIEW_SUMMARY_PATH,
        source="rrf_one_view",
    )

    print("Loading multi-view RRF/RCS summary...")
    multiview_summary = standardize_metric_summary(
        MULTIVIEW_SUMMARY_PATH,
        source="multiview",
    )

    replacement_summary_standard = replacement_summary.copy()

    metric_rows = pd.concat(
        [
            replacement_summary_standard,
            rrf_one_summary,
            multiview_summary,
        ],
        ignore_index=True,
        sort=False,
    )

    lookup = make_lookup(metric_rows)

    main_configs = [
        {
            "family": "Baseline",
            "method": "BM25 original",
            "setting": "canonical filepath + chunk query",
            "interpretation": "Canonical LegalBench-RAG mini lexical baseline.",
        },
        {
            "family": "Replacement",
            "method": "DeepSeek keyword alone",
            "method_key": "bm25_deepseek_keyword_expansion",
            "setting": "keyword_expansion replaces original",
            "interpretation": "Strong top-rank gains, but replacement remains anchor-sensitive.",
        },
        {
            "family": "Replacement",
            "method": "GPT keyword alone",
            "method_key": "bm25_gpt_keyword_expansion",
            "setting": "keyword_expansion replaces original",
            "interpretation": "Anchor loss on contract tasks; not robust as a replacement.",
        },
        {
            "family": "Replacement",
            "method": "GPT HyDE alone",
            "method_key": "bm25_gpt_hyde_style",
            "setting": "hyde_style replaces original",
            "interpretation": "Negative control: pseudo-passage reformulation dilutes document anchors.",
        },
        {
            "family": "RRF one-view",
            "method": "Original + DeepSeek keyword",
            "method_key": "rrf_bm25_original_deepseek_keyword_expansion_mini",
            "setting": "original + keyword_expansion",
            "interpretation": "Best simple one-view fusion; strong and significant top-rank gains.",
        },
        {
            "family": "RRF one-view",
            "method": "Original + GPT legal rewrite",
            "method_key": "rrf_bm25_original_gpt_legal_rewrite_mini",
            "setting": "original + legal_rewrite",
            "interpretation": "Moderate but useful top-rank gain; preserves document anchors better than GPT keyword.",
        },
        {
            "family": "RRF multi-view",
            "method": "Original + DeepSeek keyword + GPT legal",
            "method_key": "rrf_original_deepseek_keyword_gpt_legal_mini",
            "setting": "original + DeepSeek keyword_expansion + GPT legal_rewrite",
            "interpretation": "Best recall-oriented multi-view fusion; all four metrics significant after Holm.",
        },
        {
            "family": "RRF multi-view",
            "method": "Original + all non-HyDE",
            "method_key": "rrf_original_all_non_hyde_mini",
            "setting": "original + DS legal + DS keyword + GPT legal + GPT keyword",
            "interpretation": "Strong all-useful-view RRF; all metrics significant after Holm.",
        },
        {
            "family": "RCS selected",
            "method": "RCS all non-HyDE",
            "method_key": "rcs_original_all_non_hyde_mini",
            "setting": "alpha=1, beta=1, gamma=1, min_votes=2; excludes HyDE",
            "interpretation": "Best top-rank configuration; largest MRR@10 and nDCG@10 gains.",
        },
        {
            "family": "RCS control",
            "method": "RCS all views including HyDE",
            "method_key": "rcs_original_all_views_control_mini",
            "setting": "alpha=1, beta=1, gamma=1, min_votes=2; includes HyDE",
            "interpretation": "Control showing that HyDE dilutes recall/top-rank relative to non-HyDE RCS.",
        },
    ]

    full_configs = [
        main_configs[0],
        {
            "family": "Replacement",
            "method": "DeepSeek legal rewrite alone",
            "method_key": "bm25_deepseek_legal_rewrite",
            "setting": "legal_rewrite replaces original",
            "interpretation": "Small mixed effect; less useful than DeepSeek keyword.",
        },
        main_configs[1],
        {
            "family": "Replacement",
            "method": "DeepSeek HyDE alone",
            "method_key": "bm25_deepseek_hyde_style",
            "setting": "hyde_style replaces original",
            "interpretation": "Weak replacement; loses important document anchors.",
        },
        {
            "family": "Replacement",
            "method": "GPT legal rewrite alone",
            "method_key": "bm25_gpt_legal_rewrite",
            "setting": "legal_rewrite replaces original",
            "interpretation": "Best GPT replacement; preserves anchors and improves top-rank.",
        },
        main_configs[2],
        main_configs[3],
        {
            "family": "RRF one-view",
            "method": "Original + DeepSeek legal rewrite",
            "method_key": "rrf_bm25_original_deepseek_legal_rewrite_mini",
            "setting": "original + legal_rewrite",
            "interpretation": "Low-risk but mostly non-significant gains.",
        },
        main_configs[4],
        {
            "family": "RRF one-view",
            "method": "Original + DeepSeek HyDE",
            "method_key": "rrf_bm25_original_deepseek_hyde_style_mini",
            "setting": "original + hyde_style",
            "interpretation": "HyDE remains harmful for Recall@100 even with original anchoring.",
        },
        main_configs[5],
        {
            "family": "RRF one-view",
            "method": "Original + GPT keyword",
            "method_key": "rrf_bm25_original_gpt_keyword_expansion_mini",
            "setting": "original + keyword_expansion",
            "interpretation": "RRF partly repairs GPT keyword anchor loss, but not enough.",
        },
        {
            "family": "RRF one-view",
            "method": "Original + GPT HyDE",
            "method_key": "rrf_bm25_original_gpt_hyde_style_mini",
            "setting": "original + hyde_style",
            "interpretation": "Significantly harmful; strongest negative control.",
        },
        {
            "family": "RRF multi-view",
            "method": "Original + DeepSeek legal + keyword",
            "method_key": "rrf_original_deepseek_legal_keyword_mini",
            "setting": "original + DS legal + DS keyword",
            "interpretation": "Strong non-HyDE DeepSeek-only fusion.",
        },
        {
            "family": "RRF multi-view",
            "method": "Original + GPT legal + keyword",
            "method_key": "rrf_original_gpt_legal_keyword_mini",
            "setting": "original + GPT legal + GPT keyword",
            "interpretation": "GPT-only fusion improves top-rank but is weaker after Holm.",
        },
        main_configs[6],
        {
            "family": "RRF multi-view",
            "method": "Original + DS legal + DS keyword + GPT legal",
            "method_key": "rrf_original_deepseek_legal_keyword_gpt_legal_mini",
            "setting": "original + DS legal + DS keyword + GPT legal",
            "interpretation": "Strong but slightly weaker than DS keyword + GPT legal for recall.",
        },
        main_configs[7],
        {
            "family": "RRF control",
            "method": "Original + all views including HyDE",
            "method_key": "rrf_original_all_views_control_mini",
            "setting": "original + all six reformulations including HyDE",
            "interpretation": "Control: adding HyDE reduces robustness relative to non-HyDE fusion.",
        },
        {
            "family": "RCS selected",
            "method": "RCS DeepSeek legal + keyword",
            "method_key": "rcs_original_deepseek_legal_keyword_mini",
            "setting": "alpha=1, beta=1, gamma=1, min_votes=2",
            "interpretation": "Good RCS DeepSeek-only configuration.",
        },
        {
            "family": "RCS selected",
            "method": "RCS DeepSeek keyword + GPT legal",
            "method_key": "rcs_original_deepseek_keyword_gpt_legal_mini",
            "setting": "alpha=1, beta=1, gamma=1, min_votes=2",
            "interpretation": "Best RCS Recall@10 configuration.",
        },
        {
            "family": "RCS selected",
            "method": "RCS DS legal + DS keyword + GPT legal",
            "method_key": "rcs_original_deepseek_legal_keyword_gpt_legal_mini",
            "setting": "alpha=1, beta=1, gamma=1, min_votes=2",
            "interpretation": "Best RCS Recall@100 among selected configurations.",
        },
        main_configs[8],
        main_configs[9],
    ]

    main_df = build_table(
        configs=main_configs,
        lookup=lookup,
        baseline_scores=baseline_scores,
    )

    full_df = build_table(
        configs=full_configs,
        lookup=lookup,
        baseline_scores=baseline_scores,
    )

    main_df.to_csv(OUT_MAIN_CSV, index=False)
    full_df.to_csv(OUT_FULL_CSV, index=False)

    write_markdown(
        OUT_MAIN_MD,
        "LegalBench-RAG mini final synthesis table — MAIN",
        main_df,
    )

    write_markdown(
        OUT_FULL_MD,
        "LegalBench-RAG mini final synthesis table — FULL",
        full_df,
    )

    print("\n" + "=" * 80)
    print("LegalBench final synthesis table — MAIN")
    print("=" * 80)
    print(main_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("LegalBench final synthesis table — FULL")
    print("=" * 80)
    print(full_df.to_string(index=False))

    print("\nSaved files:")
    print(OUT_REPLACEMENT_SUMMARY)
    print(OUT_MAIN_CSV)
    print(OUT_FULL_CSV)
    print(OUT_MAIN_MD)
    print(OUT_FULL_MD)

    print("\nNext:")
    print("  Build the cross-dataset BSARD + LegalBench synthesis table.")


if __name__ == "__main__":
    main()