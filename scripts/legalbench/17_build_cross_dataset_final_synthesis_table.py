"""Build the cross-dataset synthesis tables for BSARD and LegalBench."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BSARD_DIR = Path("outputs/tables/bsard")
LEGALBENCH_DIR = Path("outputs/tables/legalbench")
OUT_DIR = Path("outputs/tables/cross_dataset")

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MAIN_CSV = OUT_DIR / "cross_dataset_final_synthesis_table_main.csv"
OUT_FULL_CSV = OUT_DIR / "cross_dataset_final_synthesis_table_full.csv"
OUT_KEY_FINDINGS_CSV = OUT_DIR / "cross_dataset_key_findings.csv"

OUT_MAIN_MD = OUT_DIR / "cross_dataset_final_synthesis_table_main.md"
OUT_FULL_MD = OUT_DIR / "cross_dataset_final_synthesis_table_full.md"
OUT_KEY_FINDINGS_MD = OUT_DIR / "cross_dataset_key_findings.md"


STANDARD_COLUMNS = [
    "dataset",
    "dataset_profile",
    "family",
    "method",
    "setting",
    "Recall@10",
    "Delta Recall@10",
    "pHolm Recall@10",
    "Recall@100",
    "Delta Recall@100",
    "Harm Recall@100 %",
    "pHolm Recall@100",
    "MRR@10",
    "Delta MRR@10",
    "pHolm MRR@10",
    "nDCG@10",
    "Delta nDCG@10",
    "Harm nDCG@10 %",
    "pHolm nDCG@10",
    "Holm significant positive metrics",
    "Holm significant negative metrics",
    "interpretation",
]


SCORE_COLUMNS = {
    "Recall@10",
    "Recall@100",
    "MRR@10",
    "nDCG@10",
}

DELTA_COLUMNS = {
    "Delta Recall@10",
    "Delta Recall@100",
    "Delta MRR@10",
    "Delta nDCG@10",
}

P_COLUMNS = {
    "pHolm Recall@10",
    "pHolm Recall@100",
    "pHolm MRR@10",
    "pHolm nDCG@10",
}

PERCENT_COLUMNS = {
    "Harm Recall@100 %",
    "Harm nDCG@10 %",
}


def resolve_table(
    *,
    directory: Path,
    exact_candidates: list[str],
    glob_patterns: list[str],
    label: str,
) -> Path:
    """
    Trouve une table même si le nom exact diffère légèrement.
    """
    for name in exact_candidates:
        path = directory / name

        if path.exists():
            return path

    matches = []

    for pattern in glob_patterns:
        matches.extend(directory.glob(pattern))

    matches = sorted(set(matches), key=lambda p: str(p).lower())

    if matches:
        return matches[0]

    searched = "\n".join(
        [f"  - {directory / name}" for name in exact_candidates]
        + [f"  - {directory / pattern}" for pattern in glob_patterns]
    )

    raise FileNotFoundError(
        f"Could not find {label} table.\n"
        f"Searched:\n{searched}"
    )


def read_csv_as_text(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    return df.fillna("")


def get_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    existing = {col.lower(): col for col in df.columns}

    for cand in candidates:
        key = cand.lower()

        if key in existing:
            return existing[key]

    return None


def parse_float(value) -> float | None:
    text = str(value).strip()

    if not text or text.lower() in {"nan", "none"}:
        return None

    text = text.replace("%", "").replace(",", "")

    try:
        return float(text)
    except ValueError:
        return None


def format_value(column: str, value: str) -> str:
    """
    Formate les nombres pour éviter les longues décimales dans les tables finales.
    """
    value = str(value).strip()

    if value == "":
        return ""

    numeric = parse_float(value)

    if numeric is None:
        return value

    if column in SCORE_COLUMNS:
        return f"{numeric:.6f}"

    if column in DELTA_COLUMNS:
        return f"{numeric:+.4f}"

    if column in P_COLUMNS:
        return f"{numeric:.4f}"

    if column in PERCENT_COLUMNS:
        return f"{numeric:.2f}"

    return value


def normalize_final_table(
    *,
    df: pd.DataFrame,
    dataset: str,
    dataset_profile: str,
) -> pd.DataFrame:
    """
    Harmonise les tables BSARD et LegalBench.

    BSARD et LegalBench peuvent avoir quelques différences de nommage :
      - "holm significant metrics"
      - "Holm significant positive metrics"
      - "Holm significant negative metrics"
    """
    colmap = {
        "family": get_column(df, ["family", "Family"]),
        "method": get_column(df, ["method", "Method"]),
        "setting": get_column(df, ["setting", "Setting"]),
        "Recall@10": get_column(df, ["Recall@10"]),
        "Delta Recall@10": get_column(df, ["Delta Recall@10", "delta Recall@10"]),
        "pHolm Recall@10": get_column(df, ["pHolm Recall@10", "p_holm Recall@10"]),
        "Recall@100": get_column(df, ["Recall@100"]),
        "Delta Recall@100": get_column(df, ["Delta Recall@100", "delta Recall@100"]),
        "Harm Recall@100 %": get_column(df, ["Harm Recall@100 %", "harm Recall@100 %"]),
        "pHolm Recall@100": get_column(df, ["pHolm Recall@100", "p_holm Recall@100"]),
        "MRR@10": get_column(df, ["MRR@10"]),
        "Delta MRR@10": get_column(df, ["Delta MRR@10", "delta MRR@10"]),
        "pHolm MRR@10": get_column(df, ["pHolm MRR@10", "p_holm MRR@10"]),
        "nDCG@10": get_column(df, ["nDCG@10"]),
        "Delta nDCG@10": get_column(df, ["Delta nDCG@10", "delta nDCG@10"]),
        "Harm nDCG@10 %": get_column(df, ["Harm nDCG@10 %", "harm nDCG@10 %"]),
        "pHolm nDCG@10": get_column(df, ["pHolm nDCG@10", "p_holm nDCG@10"]),
        "Holm significant positive metrics": get_column(
            df,
            [
                "Holm significant positive metrics",
                "holm significant positive metrics",
                "holm significant metrics",
                "Holm significant metrics",
            ],
        ),
        "Holm significant negative metrics": get_column(
            df,
            [
                "Holm significant negative metrics",
                "holm significant negative metrics",
            ],
        ),
        "interpretation": get_column(df, ["interpretation", "Interpretation"]),
    }

    rows = []

    for _, row in df.iterrows():
        out = {
            "dataset": dataset,
            "dataset_profile": dataset_profile,
        }

        for target_col in STANDARD_COLUMNS:
            if target_col in {"dataset", "dataset_profile"}:
                continue

            source_col = colmap.get(target_col)

            if source_col is None:
                out[target_col] = ""
            else:
                raw_value = str(row.get(source_col, "")).strip()
                out[target_col] = format_value(target_col, raw_value)

        # Si l'ancienne table BSARD a seulement "holm significant metrics",
        # on le traite comme des métriques positives par défaut.
        # Mais si une métrique significative a un delta négatif, on la déplace
        # dans la colonne "negative metrics".
        if dataset == "BSARD" and not out["Holm significant negative metrics"]:
            neg_like = []

            sig_text = out["Holm significant positive metrics"]
            delta_cols = [
                ("Recall@10", "Delta Recall@10"),
                ("Recall@100", "Delta Recall@100"),
                ("MRR@10", "Delta MRR@10"),
                ("nDCG@10", "Delta nDCG@10"),
            ]

            for metric, delta_col in delta_cols:
                delta_val = parse_float(out.get(delta_col, ""))

                if delta_val is None:
                    continue

                if metric in sig_text and delta_val < 0:
                    neg_like.append(metric)

            if neg_like:
                out["Holm significant negative metrics"] = ", ".join(neg_like)

                positive = [
                    metric.strip()
                    for metric in sig_text.split(",")
                    if metric.strip() and metric.strip() not in neg_like
                ]

                out["Holm significant positive metrics"] = ", ".join(positive)

        rows.append(out)

    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def write_markdown(path: Path, title: str, df: pd.DataFrame) -> None:
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(df.to_markdown(index=False))
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def contains_all(text: str, required_terms: list[str]) -> bool:
    lowered = str(text).lower()
    return all(term.lower() in lowered for term in required_terms)


def contains_any(text: str, terms: list[str]) -> bool:
    lowered = str(text).lower()
    return any(term.lower() in lowered for term in terms)


def find_row(
    df: pd.DataFrame,
    *,
    dataset: str,
    family_terms: list[str] | None = None,
    method_terms: list[str] | None = None,
    setting_terms: list[str] | None = None,
    family_exclude_terms: list[str] | None = None,
    method_exclude_terms: list[str] | None = None,
    setting_exclude_terms: list[str] | None = None,
) -> pd.Series | None:
    subset = df[df["dataset"].str.lower() == dataset.lower()].copy()

    for _, row in subset.iterrows():
        family = str(row.get("family", ""))
        method = str(row.get("method", ""))
        setting = str(row.get("setting", ""))

        if family_terms and not contains_all(family, family_terms):
            continue

        if method_terms and not contains_all(method, method_terms):
            continue

        if setting_terms and not contains_all(setting, setting_terms):
            continue

        if family_exclude_terms and contains_any(family, family_exclude_terms):
            continue

        if method_exclude_terms and contains_any(method, method_exclude_terms):
            continue

        if setting_exclude_terms and contains_any(setting, setting_exclude_terms):
            continue

        return row

    return None


def first_non_none(*items):
    """
    Retourne le premier objet non None.

    Important :
    On n'utilise pas `a or b` avec pandas.Series, car une Series
    n'a pas de vérité booléenne unique.
    """
    for item in items:
        if item is not None:
            return item

    return None


def metric_snapshot(row: pd.Series | None) -> str:
    if row is None:
        return ""

    pieces = []

    for metric in ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]:
        delta_col = f"Delta {metric}"
        p_col = f"pHolm {metric}"

        score = str(row.get(metric, "")).strip()
        delta = str(row.get(delta_col, "")).strip()
        p_holm = str(row.get(p_col, "")).strip()

        if score:
            if delta and p_holm:
                pieces.append(f"{metric}={score} ({delta}, pHolm={p_holm})")
            elif delta:
                pieces.append(f"{metric}={score} ({delta})")
            else:
                pieces.append(f"{metric}={score}")

    return "; ".join(pieces)


def build_key_findings(main_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit une table narrative courte pour l'introduction/résultats.
    """
    bsard_keyword = find_row(
        main_df,
        dataset="BSARD",
        family_terms=["RRF"],
        method_terms=["deepseek", "keyword"],
    )

    bsard_rcs = first_non_none(
        find_row(
            main_df,
            dataset="BSARD",
            family_terms=["RCS"],
            method_terms=["strong", "consensus"],
        ),
        find_row(
            main_df,
            dataset="BSARD",
            family_terms=["RCS"],
            method_terms=["top"],
        ),
        find_row(
            main_df,
            dataset="BSARD",
            family_terms=["RCS"],
            method_terms=["all"],
        ),
        find_row(
            main_df,
            dataset="BSARD",
            family_terms=["RCS"],
        ),
    )

    legal_gpt_keyword = find_row(
        main_df,
        dataset="LegalBench-RAG mini",
        family_terms=["Replacement"],
        method_terms=["gpt", "keyword"],
    )

    legal_rrf_best = find_row(
        main_df,
        dataset="LegalBench-RAG mini",
        family_terms=["RRF", "multi"],
        method_terms=["deepseek", "keyword", "gpt", "legal"],
    )

    legal_rcs_non_hyde = find_row(
        main_df,
        dataset="LegalBench-RAG mini",
        family_terms=["RCS"],
        method_terms=["non-hyde"],
    )

    # Attention :
    # "non-HyDE" contient aussi "HyDE".
    # On force donc "including HyDE" et on exclut "non-HyDE".
    legal_hyde_control = first_non_none(
        find_row(
            main_df,
            dataset="LegalBench-RAG mini",
            family_terms=["RCS", "control"],
            method_terms=["including", "HyDE"],
            method_exclude_terms=["non-HyDE"],
        ),
        find_row(
            main_df,
            dataset="LegalBench-RAG mini",
            family_terms=["RCS"],
            method_terms=["all", "views", "including", "HyDE"],
            method_exclude_terms=["non-HyDE"],
        ),
        find_row(
            main_df,
            dataset="LegalBench-RAG mini",
            family_terms=["RCS"],
            setting_terms=["includes", "HyDE"],
            method_exclude_terms=["non-HyDE"],
            setting_exclude_terms=["excludes", "non-HyDE"],
        ),
    )

    rows = [
        {
            "finding_id": "F1",
            "theme": "Original-preserving fusion generalizes",
            "dataset": "BSARD",
            "evidence_method": "" if bsard_keyword is None else bsard_keyword.get("method", ""),
            "evidence": metric_snapshot(bsard_keyword),
            "interpretation": (
                "BSARD supports the core pattern that reformulations are safer "
                "as auxiliary retrieval views than as replacements."
            ),
        },
        {
            "finding_id": "F2",
            "theme": "Consensus improves top-rank robustness",
            "dataset": "BSARD",
            "evidence_method": "" if bsard_rcs is None else bsard_rcs.get("method", ""),
            "evidence": metric_snapshot(bsard_rcs),
            "interpretation": (
                "The BSARD RCS results motivate consensus-aware retrieval scoring."
            ),
        },
        {
            "finding_id": "F3",
            "theme": "Replacement can fail through anchor loss",
            "dataset": "LegalBench-RAG mini",
            "evidence_method": "" if legal_gpt_keyword is None else legal_gpt_keyword.get("method", ""),
            "evidence": metric_snapshot(legal_gpt_keyword),
            "interpretation": (
                "LegalBench shows the mechanism behind replacement failures: "
                "some reformulations delete document, party, or file-path anchors."
            ),
        },
        {
            "finding_id": "F4",
            "theme": "Best LegalBench recall-oriented configuration",
            "dataset": "LegalBench-RAG mini",
            "evidence_method": "" if legal_rrf_best is None else legal_rrf_best.get("method", ""),
            "evidence": metric_snapshot(legal_rrf_best),
            "interpretation": (
                "RRF with DeepSeek keyword expansion and GPT legal rewrite gives "
                "the strongest broad-recall LegalBench configuration."
            ),
        },
        {
            "finding_id": "F5",
            "theme": "Best LegalBench top-rank configuration",
            "dataset": "LegalBench-RAG mini",
            "evidence_method": "" if legal_rcs_non_hyde is None else legal_rcs_non_hyde.get("method", ""),
            "evidence": metric_snapshot(legal_rcs_non_hyde),
            "interpretation": (
                "RCS over non-HyDE views gives the largest top-rank gains, "
                "especially for MRR@10 and nDCG@10."
            ),
        },
        {
            "finding_id": "F6",
            "theme": "HyDE is task-dependent and harmful on document-centric RAG",
            "dataset": "LegalBench-RAG mini",
            "evidence_method": "" if legal_hyde_control is None else legal_hyde_control.get("method", ""),
            "evidence": metric_snapshot(legal_hyde_control),
            "interpretation": (
                "The HyDE control confirms that adding pseudo-passage views reduces "
                "robustness relative to the non-HyDE RCS configuration, especially "
                "for Recall@100 and nDCG@10."
            ),
        },
        {
            "finding_id": "F7",
            "theme": "Cross-dataset conclusion",
            "dataset": "BSARD + LegalBench-RAG mini",
            "evidence_method": "",
            "evidence": "",
            "interpretation": (
                "Across both datasets, the safest strategy is not to replace the "
                "original query, but to preserve it as an anchor and use selected "
                "LLM reformulations as auxiliary retrieval views."
            ),
        },
    ]

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cross-dataset final synthesis tables."
    )

    parser.add_argument(
        "--bsard-main",
        type=Path,
        default=None,
        help="Optional explicit path to BSARD main synthesis CSV.",
    )

    parser.add_argument(
        "--bsard-full",
        type=Path,
        default=None,
        help="Optional explicit path to BSARD full synthesis CSV.",
    )

    parser.add_argument(
        "--legalbench-main",
        type=Path,
        default=LEGALBENCH_DIR / "legalbench_final_synthesis_table_main.csv",
        help="Path to LegalBench main synthesis CSV.",
    )

    parser.add_argument(
        "--legalbench-full",
        type=Path,
        default=LEGALBENCH_DIR / "legalbench_final_synthesis_table_full.csv",
        help="Path to LegalBench full synthesis CSV.",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Building cross-dataset final synthesis tables")
    print("=" * 80)

    if args.bsard_main is not None:
        bsard_main_path = args.bsard_main
    else:
        bsard_main_path = resolve_table(
            directory=BSARD_DIR,
            exact_candidates=[
                "bm25_final_synthesis_table_main.csv",
                "bsard_final_synthesis_table_main.csv",
                "bm25_final_synthesis_main.csv",
            ],
            glob_patterns=[
                "*final*synthesis*main*.csv",
                "*synthesis*main*.csv",
            ],
            label="BSARD MAIN",
        )

    if args.bsard_full is not None:
        bsard_full_path = args.bsard_full
    else:
        bsard_full_path = resolve_table(
            directory=BSARD_DIR,
            exact_candidates=[
                "bm25_final_synthesis_table_full.csv",
                "bsard_final_synthesis_table_full.csv",
                "bm25_final_synthesis_full.csv",
            ],
            glob_patterns=[
                "*final*synthesis*full*.csv",
                "*synthesis*full*.csv",
            ],
            label="BSARD FULL",
        )

    legal_main_path = args.legalbench_main
    legal_full_path = args.legalbench_full

    for path in [bsard_main_path, bsard_full_path, legal_main_path, legal_full_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required table: {path}")

    print(f"BSARD main:      {bsard_main_path}")
    print(f"BSARD full:      {bsard_full_path}")
    print(f"LegalBench main: {legal_main_path}")
    print(f"LegalBench full: {legal_full_path}")

    bsard_main = normalize_final_table(
        df=read_csv_as_text(bsard_main_path),
        dataset="BSARD",
        dataset_profile="Statutory legal retrieval; Belgian law queries.",
    )

    bsard_full = normalize_final_table(
        df=read_csv_as_text(bsard_full_path),
        dataset="BSARD",
        dataset_profile="Statutory legal retrieval; Belgian law queries.",
    )

    legal_main = normalize_final_table(
        df=read_csv_as_text(legal_main_path),
        dataset="LegalBench-RAG mini",
        dataset_profile="Document-centric legal RAG; contracts and privacy policies.",
    )

    legal_full = normalize_final_table(
        df=read_csv_as_text(legal_full_path),
        dataset="LegalBench-RAG mini",
        dataset_profile="Document-centric legal RAG; contracts and privacy policies.",
    )

    cross_main = pd.concat([bsard_main, legal_main], ignore_index=True)
    cross_full = pd.concat([bsard_full, legal_full], ignore_index=True)

    key_findings = build_key_findings(cross_main)

    cross_main.to_csv(OUT_MAIN_CSV, index=False)
    cross_full.to_csv(OUT_FULL_CSV, index=False)
    key_findings.to_csv(OUT_KEY_FINDINGS_CSV, index=False)

    write_markdown(
        OUT_MAIN_MD,
        "Cross-dataset final synthesis table — MAIN",
        cross_main,
    )

    write_markdown(
        OUT_FULL_MD,
        "Cross-dataset final synthesis table — FULL",
        cross_full,
    )

    write_markdown(
        OUT_KEY_FINDINGS_MD,
        "Cross-dataset key findings",
        key_findings,
    )

    print("\n" + "=" * 80)
    print("Cross-dataset final synthesis table — MAIN")
    print("=" * 80)
    print(cross_main.to_string(index=False))

    print("\n" + "=" * 80)
    print("Cross-dataset key findings")
    print("=" * 80)
    print(key_findings.to_string(index=False))

    print("\nSaved files:")
    print(OUT_MAIN_CSV)
    print(OUT_FULL_CSV)
    print(OUT_KEY_FINDINGS_CSV)
    print(OUT_MAIN_MD)
    print(OUT_FULL_MD)
    print(OUT_KEY_FINDINGS_MD)

    print("\nNext:")
    print("  Use cross_dataset_key_findings.md to write the paper-level Results summary.")


if __name__ == "__main__":
    main()