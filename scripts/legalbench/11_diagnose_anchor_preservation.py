"""Measure source-anchor preservation in LegalBench reformulations."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/processed/legalbench/rag_mini")
OUT_DIR = Path("outputs/tables/legalbench")
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUERIES_PATH = DATA_DIR / "queries.jsonl"
QRELS_DETAIL_PATH = DATA_DIR / "qrels_detail.tsv"
REFORMULATIONS_PATH = (
    DATA_DIR / "reformulations/normalized/all_generators_mini.jsonl"
)

BASELINE_PER_QUERY_PATH = OUT_DIR / "bm25_original_mini_canonical_per_query.csv"
REFORM_PER_QUERY_PATH = OUT_DIR / "bm25_reformulations_mini_per_query_all.csv"

OUT_PER_QUERY = OUT_DIR / "anchor_preservation_per_query.csv"
OUT_SUMMARY = OUT_DIR / "anchor_preservation_summary.csv"
OUT_CORRELATIONS = OUT_DIR / "anchor_preservation_metric_correlations.csv"
OUT_EXAMPLES = OUT_DIR / "anchor_loss_examples.csv"
OUT_MD = OUT_DIR / "anchor_preservation_summary.md"


QUERY_TYPES = [
    "legal_rewrite",
    "keyword_expansion",
    "hyde_style",
]

METRICS = [
    "Recall@10",
    "Recall@100",
    "MRR@10",
    "nDCG@10",
]


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "can",
    "clause", "confidential", "confidentiality", "consider", "contract",
    "document", "does", "for", "from", "has", "have", "if", "in", "include",
    "includes", "information", "is", "it", "may", "of", "on", "or", "party",
    "policy", "privacy", "receiving", "shall", "some", "that", "the", "their",
    "this", "to", "under", "use", "what", "whether", "which", "who", "with",
    "agreement", "agreements", "non", "disclosure", "mutual", "nda",
    "txt", "pdf", "inc", "llc", "ltd", "corp", "corporation", "company",
    "companies", "systems", "group", "holdings", "limited",
    "contractnli", "cuad", "maud", "privacy", "qa", "privacy_qa",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {path}: {exc}"
                ) from exc

    return rows


def tokenize(text: str) -> list[str]:
    """
    Tokenisation simple, en minuscules.
    """
    text = (text or "").lower()
    return re.findall(r"\b[a-z0-9]+\b", text)


def content_tokens(text: str) -> set[str]:
    """
    Tokens utiles, sans stopwords et sans tokens trop courts.
    """
    return {
        token
        for token in tokenize(text)
        if len(token) >= 3 and token not in STOPWORDS
    }


def clean_path_text(path: str) -> str:
    """
    Transforme un file_path en texte tokenisable.
    """
    text = path.replace("\\", "/")
    text = text.replace("/", " ")
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("||", " ")
    text = text.replace(".txt", " ")
    text = text.replace(".pdf", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_capitalized_tokens(text: str) -> set[str]:
    """
    Extrait des tokens qui ressemblent à des noms propres ou identifiants.

    Exemples :
      DoiT, ICN, Fiverr, Apollo, FullStory, M5-Systems
    """
    raw_tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9'\-]*\b", text or "")

    result = set()

    for tok in raw_tokens:
        clean = tok.strip("'").strip("-")

        if len(clean) < 2:
            continue

        lower = clean.lower()

        if lower in STOPWORDS:
            continue

        # Garde si majuscule interne, acronyme, ou initiale majuscule.
        has_upper = any(ch.isupper() for ch in clean)
        is_acronym = clean.isupper() and len(clean) >= 2
        has_digit = any(ch.isdigit() for ch in clean)

        if has_upper or is_acronym or has_digit:
            result.add(lower)

    return result


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return np.nan
    return numerator / denominator


def build_query_maps() -> tuple[dict[str, dict], dict[str, set[str]], dict[str, set[str]]]:
    """
    Retourne :
      query_id -> query row
      query_id -> anchor tokens depuis les file_path gold
      query_id -> file_path strings
    """
    queries = read_jsonl(QUERIES_PATH)
    query_map = {str(row["query_id"]): row for row in queries}

    qrels_detail = pd.read_csv(QRELS_DETAIL_PATH, sep="\t")

    file_paths_by_qid: dict[str, set[str]] = {}
    anchor_tokens_by_qid: dict[str, set[str]] = {}

    for qid, group in qrels_detail.groupby("query_id"):
        paths = set(str(x) for x in group["file_path"].dropna().unique())
        file_paths_by_qid[str(qid)] = paths

        anchor_tokens = set()

        for path in paths:
            anchor_tokens.update(content_tokens(clean_path_text(path)))

        anchor_tokens_by_qid[str(qid)] = anchor_tokens

    return query_map, anchor_tokens_by_qid, file_paths_by_qid


def load_metric_gains() -> pd.DataFrame:
    """
    Charge les métriques per-query des reformulations et calcule
    les gains par rapport à la baseline canonique.
    """
    baseline = pd.read_csv(BASELINE_PER_QUERY_PATH)
    reform = pd.read_csv(REFORM_PER_QUERY_PATH)

    baseline_cols = ["query_id", *METRICS]

    baseline = baseline[baseline_cols].copy()
    baseline = baseline.rename(
        columns={metric: f"baseline_{metric}" for metric in METRICS}
    )

    merged = reform.merge(baseline, on="query_id", how="left")

    for metric in METRICS:
        merged[f"delta_{metric}"] = (
            merged[metric] - merged[f"baseline_{metric}"]
        )

    keep_cols = [
        "query_id",
        "method",
        "generator",
        "query_type",
        "task",
        *METRICS,
        *[f"baseline_{metric}" for metric in METRICS],
        *[f"delta_{metric}" for metric in METRICS],
    ]

    return merged[keep_cols].copy()


def make_anchor_diagnostic_rows() -> pd.DataFrame:
    query_map, anchor_tokens_by_qid, file_paths_by_qid = build_query_maps()
    reform_rows = read_jsonl(REFORMULATIONS_PATH)
    metric_gains = load_metric_gains()

    rows = []

    for row in reform_rows:
        qid = str(row["query_id"])
        generator = str(row["generator"])
        task = str(row["task"])

        source_query = query_map[qid]
        original_text = str(source_query.get("text") or source_query.get("original_text") or "")
        original_tokens = content_tokens(original_text)
        capitalized_tokens = extract_capitalized_tokens(original_text)

        anchor_tokens = anchor_tokens_by_qid.get(qid, set())
        file_paths = file_paths_by_qid.get(qid, set())

        for query_type in QUERY_TYPES:
            reform_text = str(row.get(query_type, "") or "")
            reform_tokens = content_tokens(reform_text)

            kept_original = original_tokens & reform_tokens
            dropped_original = original_tokens - reform_tokens

            kept_anchor = anchor_tokens & reform_tokens
            dropped_anchor = anchor_tokens - reform_tokens

            kept_caps = capitalized_tokens & reform_tokens
            dropped_caps = capitalized_tokens - reform_tokens

            rows.append(
                {
                    "query_id": qid,
                    "task": task,
                    "generator": generator,
                    "query_type": query_type,
                    "original_text": original_text,
                    "reformulation_text": reform_text,
                    "file_paths": " | ".join(sorted(file_paths)),
                    "original_token_count": len(original_tokens),
                    "reformulation_token_count": len(reform_tokens),
                    "anchor_token_count": len(anchor_tokens),
                    "capitalized_token_count": len(capitalized_tokens),
                    "kept_original_token_count": len(kept_original),
                    "kept_anchor_token_count": len(kept_anchor),
                    "kept_capitalized_token_count": len(kept_caps),
                    "original_token_retention": safe_ratio(
                        len(kept_original), len(original_tokens)
                    ),
                    "anchor_token_retention": safe_ratio(
                        len(kept_anchor), len(anchor_tokens)
                    ),
                    "capitalized_token_retention": safe_ratio(
                        len(kept_caps), len(capitalized_tokens)
                    ),
                    "kept_original_tokens": ", ".join(sorted(kept_original)),
                    "dropped_original_tokens": ", ".join(sorted(dropped_original)),
                    "kept_anchor_tokens": ", ".join(sorted(kept_anchor)),
                    "dropped_anchor_tokens": ", ".join(sorted(dropped_anchor)),
                    "kept_capitalized_tokens": ", ".join(sorted(kept_caps)),
                    "dropped_capitalized_tokens": ", ".join(sorted(dropped_caps)),
                }
            )

    diag = pd.DataFrame(rows)

    diag = diag.merge(
        metric_gains,
        on=["query_id", "task", "generator", "query_type"],
        how="left",
    )

    return diag


def make_summary(diag: pd.DataFrame) -> pd.DataFrame:
    summary = (
        diag
        .groupby(["generator", "query_type", "task"])
        .agg(
            num_queries=("query_id", "count"),
            avg_original_token_retention=("original_token_retention", "mean"),
            avg_anchor_token_retention=("anchor_token_retention", "mean"),
            avg_capitalized_token_retention=("capitalized_token_retention", "mean"),
            zero_anchor_retention_rate=(
                "anchor_token_retention",
                lambda x: float((x.fillna(0) == 0).mean()),
            ),
            avg_reformulation_token_count=("reformulation_token_count", "mean"),
            delta_Recall_at_10=("delta_Recall@10", "mean"),
            delta_Recall_at_100=("delta_Recall@100", "mean"),
            delta_MRR_at_10=("delta_MRR@10", "mean"),
            delta_nDCG_at_10=("delta_nDCG@10", "mean"),
        )
        .reset_index()
    )

    return summary


def make_correlations(diag: pd.DataFrame) -> pd.DataFrame:
    rows = []

    retention_cols = [
        "original_token_retention",
        "anchor_token_retention",
        "capitalized_token_retention",
    ]

    delta_cols = [f"delta_{metric}" for metric in METRICS]

    for (generator, query_type, task), group in diag.groupby(
        ["generator", "query_type", "task"]
    ):
        for retention_col in retention_cols:
            for delta_col in delta_cols:
                valid = group[[retention_col, delta_col]].dropna()

                if len(valid) < 3:
                    corr = np.nan
                elif valid[retention_col].nunique() <= 1:
                    corr = np.nan
                elif valid[delta_col].nunique() <= 1:
                    corr = np.nan
                else:
                    corr = float(valid[retention_col].corr(valid[delta_col]))

                rows.append(
                    {
                        "generator": generator,
                        "query_type": query_type,
                        "task": task,
                        "retention_signal": retention_col,
                        "metric_delta": delta_col,
                        "pearson_corr": corr,
                        "num_queries": len(valid),
                    }
                )

    return pd.DataFrame(rows)


def make_anchor_loss_examples(diag: pd.DataFrame) -> pd.DataFrame:
    """
    Exemples de fortes pertes de performance avec faible conservation d'ancres.
    """
    examples = diag.copy()

    examples["anchor_token_retention_filled"] = examples[
        "anchor_token_retention"
    ].fillna(0)

    examples = examples.sort_values(
        [
            "delta_Recall@10",
            "delta_nDCG@10",
            "anchor_token_retention_filled",
        ],
        ascending=[True, True, True],
    )

    selected_cols = [
        "query_id",
        "task",
        "generator",
        "query_type",
        "delta_Recall@10",
        "delta_Recall@100",
        "delta_MRR@10",
        "delta_nDCG@10",
        "original_token_retention",
        "anchor_token_retention",
        "capitalized_token_retention",
        "file_paths",
        "original_text",
        "reformulation_text",
        "kept_anchor_tokens",
        "dropped_anchor_tokens",
        "kept_capitalized_tokens",
        "dropped_capitalized_tokens",
    ]

    return examples[selected_cols].head(100)


def write_markdown_summary(
    summary: pd.DataFrame,
    examples: pd.DataFrame,
) -> None:
    lines = []

    lines.append("# LegalBench-RAG mini — Anchor preservation diagnostic")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "This diagnostic measures whether each reformulation preserves "
        "document-level anchors from the original query and from the gold "
        "file path."
    )
    lines.append("")
    lines.append("Key columns:")
    lines.append("")
    lines.append("- `avg_original_token_retention`: fraction of useful original query tokens retained.")
    lines.append("- `avg_anchor_token_retention`: fraction of gold file-path tokens retained.")
    lines.append("- `avg_capitalized_token_retention`: fraction of proper-name-like original tokens retained.")
    lines.append("- `zero_anchor_retention_rate`: share of queries where no file-path anchor token is retained.")
    lines.append("")
    lines.append("## Aggregated table")
    lines.append("")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Worst anchor-loss examples")
    lines.append("")
    preview = examples.head(20).copy()

    for col in ["original_text", "reformulation_text", "file_paths"]:
        if col in preview.columns:
            preview[col] = preview[col].astype(str).str.slice(0, 180)

    lines.append(preview.to_markdown(index=False))
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    required_files = [
        QUERIES_PATH,
        QRELS_DETAIL_PATH,
        REFORMULATIONS_PATH,
        BASELINE_PER_QUERY_PATH,
        REFORM_PER_QUERY_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    print("=" * 80)
    print("Diagnosing anchor preservation — LegalBench-RAG mini")
    print("=" * 80)

    print("Loading and computing per-query diagnostics...")
    diag = make_anchor_diagnostic_rows()

    print("Building summary...")
    summary = make_summary(diag)

    print("Computing correlations...")
    correlations = make_correlations(diag)

    print("Selecting examples...")
    examples = make_anchor_loss_examples(diag)

    diag.to_csv(OUT_PER_QUERY, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    correlations.to_csv(OUT_CORRELATIONS, index=False)
    examples.to_csv(OUT_EXAMPLES, index=False)

    write_markdown_summary(summary, examples)

    print("\n" + "=" * 80)
    print("Anchor preservation summary")
    print("=" * 80)
    print(summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("Worst anchor-loss examples")
    print("=" * 80)

    preview_cols = [
        "query_id",
        "task",
        "generator",
        "query_type",
        "delta_Recall@10",
        "delta_Recall@100",
        "anchor_token_retention",
        "capitalized_token_retention",
        "dropped_anchor_tokens",
    ]

    print(examples[preview_cols].head(20).to_string(index=False))

    print("\nSaved files:")
    print(OUT_PER_QUERY)
    print(OUT_SUMMARY)
    print(OUT_CORRELATIONS)
    print(OUT_EXAMPLES)
    print(OUT_MD)

    print("\nNext:")
    print("  scripts/legalbench/12_rrf_original_reformulation.py")


if __name__ == "__main__":
    main()